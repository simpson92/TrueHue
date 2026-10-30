# app/main.py
# 목적: TrueHue 웹 서버 (FastAPI) — 사진을 고르거나 올리고, 클릭한 곳의 색 이름을 알려 준다 (D13)
#   실행 (레포 폴더에서):  uvicorn app.main:app --port 8000      → 브라우저에서 http://localhost:8000
#   계산은 모두 app/engine.py (평가와 같은 부품)를 부른다. 여기서는 주소(API)와 화면 파일만 연결한다
#
#   GET  /                        화면 (app/static/index.html)
#   GET  /api/samples             ① 데이터셋 모드 사진 목록 (data/에 Gehler가 있을 때)
#   GET  /api/sample/{name}/image ?view=baseline|corrected  화면용 그림
#   POST /api/upload              ② 내 사진 올리기 → upload_id (서버 메모리에만 잠깐 보관, 디스크에 저장 안 함)
#   GET  /api/upload/{id}/image   ?view=before|after  올린 사진 (줄인 것) · 보정 후 그림
#   POST /api/analyze             {sample 또는 upload_id, x, y (normalized면 0~1 비율)} → status · color_name · baseline_name · estimated_illuminant …
#   POST /api/live                {camera: 폰 주소(예 192.168.0.12:8080), x, y (0~1 비율), white(선택)} → 폰에서 한 장 받아 분석 + 그 프레임(JPEG)
#   POST /api/live/white          {camera, x, y} → 그 자리의 흰 종이를 재서 기준(white) 또는 거절 이유. 기준은 화면이 들고 있다가 /api/live에 보냄
import base64
import ipaddress
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from app import engine

app = FastAPI(title="TrueHue", description="조명에 속지 않는 색 이름 감별기")
STATIC = Path(__file__).parent / "static"
UPLOADS = OrderedDict()            # upload_id → (sRGB 배열, 추정 조명). 최근 20장만
MAX_UPLOADS = 20
MAX_BYTES = 25 * 1024 * 1024
LIVE_TIMEOUT = 3                   # 폰이 3초 안에 한 장을 못 주면 연결 실패로 본다
LIVE_ILLUM = {}                    # 폰 주소 → 지금까지 부드럽게 한 조명 (깜빡임 줄이기, engine.smooth_illum)
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 폰은 같은 와이파이 → 프록시 설정이 있어도 거치지 않음


class AnalyzeIn(BaseModel):
    x: float
    y: float
    sample: str | None = None
    upload_id: str | None = None
    normalized: bool = False       # True면 x, y가 사진 크기에 대한 비율(0~1) — 화면은 이 방식으로 보냄


class WhiteRef(BaseModel):
    ref: list[float]               # 흰 종이의 선형 RGB (measure_white가 준 값)
    sog: list[float]               # 맞출 때의 장면 자동 추정 (조명이 바뀌었는지 비교용)


class LiveIn(BaseModel):
    camera: str                    # IP Webcam 화면 아래에 나오는 주소 (예: 192.168.0.12:8080 또는 http://192.168.0.12:8080)
    x: float = 0.5                 # 볼 곳 (프레임 너비 · 높이에 대한 비율, 기본 = 가운데)
    y: float = 0.5
    white: WhiteRef | None = None  # 흰 종이 기준 (없으면 Shades of Gray)
    view: str = "before"           # 돌려줄 프레임: before = 찍힌 그대로 · after = 보정한 그림


def live_url(camera):
    """사용자가 넣은 폰 주소 → http://<IP>:<포트>/shot.jpg

    서버가 대신 접속하는 주소이므로 같은 네트워크(사설 IP · 이 PC)만 허용한다. 경로는 항상 /shot.jpg로 고정
    """
    text = camera.strip()
    if "://" not in text:
        text = "http://" + text
    u = urllib.parse.urlsplit(text)
    if u.scheme != "http" or not u.hostname:
        raise HTTPException(400, "주소는 IP Webcam 화면 아래의 'http://숫자:8080' 형태로 넣어 주세요")
    try:
        ip = ipaddress.ip_address(u.hostname)
    except ValueError:
        raise HTTPException(400, "주소는 숫자 IP로 넣어 주세요 (예: 192.168.0.12:8080)")
    if not (ip.is_private or ip.is_loopback):
        raise HTTPException(400, "같은 와이파이 안의 주소(192.168… · 10… · 172.16~31…)만 쓸 수 있어요")
    return f"http://{ip}:{u.port or 8080}/shot.jpg"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/samples")
def samples():
    try:
        return {"available": True, "samples": engine.list_samples()}
    except FileNotFoundError:
        return {"available": False, "samples": [], "message": "data/ 또는 outputs/gehler_index.csv가 없어 데이터셋 모드를 쓸 수 없어요"}


@app.get("/api/sample/{name}/image")
def sample_image(name: str, view: str = "baseline"):
    if view not in ("baseline", "corrected"):
        raise HTTPException(400, "view는 baseline 또는 corrected")
    try:
        return Response(engine.sample_image(name, view), media_type="image/png")
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(404, str(e))


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "사진이 너무 큽니다 (25MB 이하)")
    try:
        img = engine.load_upload(data)
    except Exception:
        raise HTTPException(400, "JPG 또는 PNG 사진만 올릴 수 있어요 (아이폰 HEIC는 '호환성 우선'으로 찍거나 JPG로 바꿔 주세요)")
    uid = uuid.uuid4().hex
    UPLOADS[uid] = (img, engine.upload_illum(img))
    while len(UPLOADS) > MAX_UPLOADS:
        UPLOADS.popitem(last=False)
    h, w = img.shape[:2]
    return {"upload_id": uid, "width": w, "height": h}


@app.get("/api/upload/{uid}/image")
def upload_image(uid: str, view: str = "before"):
    if view not in ("before", "after"):
        raise HTTPException(400, "view는 before 또는 after")
    if uid not in UPLOADS:
        raise HTTPException(404, "사진이 없어요 (서버를 다시 켰다면 다시 올려 주세요)")
    img, illum = UPLOADS[uid]
    if view == "after":
        img = engine.correct_image(img, illum)
    return Response(engine.to_jpeg(img, quality=90), media_type="image/jpeg")


@app.post("/api/analyze")
def analyze(req: AnalyzeIn):
    try:
        if req.sample:
            return engine.analyze_sample(req.sample, req.x, req.y, req.normalized)
        if req.upload_id:
            if req.upload_id not in UPLOADS:
                raise HTTPException(404, "사진이 없어요 (다시 올려 주세요)")
            img, illum = UPLOADS[req.upload_id]
            return engine.analyze_upload(img, illum, req.x, req.y, req.normalized)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
    raise HTTPException(400, "sample 또는 upload_id 중 하나가 필요해요")


def fetch_frame(camera):
    """폰에서 지금 한 장 → (8비트 sRGB 배열, 받는 데 걸린 ms)"""
    url = live_url(camera)
    t0 = time.perf_counter()
    try:
        with NO_PROXY.open(url, timeout=LIVE_TIMEOUT) as r:
            data = r.read(MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError):
        raise HTTPException(502, "폰 카메라에 연결할 수 없어요 — 주소, 같은 와이파이인지, IP Webcam에서 '서버 시작'을 눌렀는지 확인해 주세요")
    ms = round(1000 * (time.perf_counter() - t0))
    try:
        return engine.load_frame(data), ms
    except Exception:
        raise HTTPException(502, "폰에서 받은 것이 사진이 아니에요 — 주소 끝의 포트(보통 8080)를 확인해 주세요")


def check_white(w):
    if w is None:
        return None
    vals = w.ref + w.sog
    if len(w.ref) != 3 or len(w.sog) != 3 or not all(v > 0 and v < 10 for v in vals):
        raise HTTPException(400, "흰 종이 기준 값이 이상해요 — '기준 지우기' 후 다시 맞춰 주세요")
    return {"ref": w.ref, "sog": w.sog}


@app.post("/api/live")
def live(req: LiveIn):
    """③ 실시간 카메라: 폰에서 지금 한 장을 받아 ② 내 사진 모드와 같은 처리 → 결과 + 그 프레임

    화면이 이 주소를 계속 불러서(한 번 끝나면 다음 요청) 실시간처럼 보인다. 프레임은 저장하지 않는다
    """
    white = check_white(req.white)
    img, fetch_ms = fetch_frame(req.camera)
    t1 = time.perf_counter()
    try:
        illum = engine.smooth_illum(LIVE_ILLUM.get(req.camera), engine.upload_illum(img))
        if illum is None:
            LIVE_ILLUM.pop(req.camera, None)
        else:
            LIVE_ILLUM[req.camera] = illum
        res = engine.analyze_live(img, req.x, req.y, white=white, illum=illum)
    except ValueError as e:
        raise HTTPException(400, str(e))
    analyze_ms = round(1000 * (time.perf_counter() - t1))
    shown = engine.correct_image(img, illum, white) if req.view == "after" else img
    frame = "data:image/jpeg;base64," + base64.b64encode(engine.to_jpeg(shown)).decode()
    return {**res, "frame": frame, "timing_ms": {"fetch": fetch_ms, "analyze": analyze_ms}}


@app.post("/api/live/white")
def live_white(req: LiveIn):
    """③ 흰 종이로 맞추기: 지금 한 장에서 (x, y) 둘레 41×41 영역을 재서 기준 또는 거절 이유 (안전장치는 engine.measure_white)"""
    img, _ = fetch_frame(req.camera)
    try:
        return engine.measure_white(img, req.x, req.y)
    except ValueError as e:
        raise HTTPException(400, str(e))