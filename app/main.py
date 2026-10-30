# app/main.py
# 목적: TrueHue 웹 서버 (FastAPI) — 사진을 고르거나 올리고, 클릭한 곳의 색 이름을 알려 준다 (D13)
#   실행 (레포 폴더에서):  uvicorn app.main:app --port 8000      → 브라우저에서 http://localhost:8000
#   계산은 모두 app/engine.py (평가와 같은 부품)를 부른다. 여기서는 주소(API)와 화면 파일만 연결한다
#
#   GET  /                        화면 (app/static/index.html)
#   GET  /api/samples             ① 데이터셋 모드 사진 목록 (data/에 Gehler가 있을 때)
#   GET  /api/sample/{name}/image ?view=baseline|corrected  화면용 그림
#   POST /api/upload              ② 내 사진 올리기 → upload_id (서버 메모리에만 잠깐 보관, 디스크에 저장 안 함)
#   GET  /api/upload/{id}/image   올린 사진 (줄인 것)
#   POST /api/analyze             {sample 또는 upload_id, x, y (normalized면 0~1 비율)} → status · color_name · baseline_name · estimated_illuminant …
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


class AnalyzeIn(BaseModel):
    x: float
    y: float
    sample: str | None = None
    upload_id: str | None = None
    normalized: bool = False       # True면 x, y가 사진 크기에 대한 비율(0~1) — 화면은 이 방식으로 보냄


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
def upload_image(uid: str):
    if uid not in UPLOADS:
        raise HTTPException(404, "사진이 없어요 (서버를 다시 켰다면 다시 올려 주세요)")
    return Response(engine.to_png(UPLOADS[uid][0]), media_type="image/png")


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
