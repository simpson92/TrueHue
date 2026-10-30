# app/engine.py
# 목적: 웹 화면의 계산 부품 — 사진 한 장과 클릭한 점 → 색 이름 (D13)
#   - 평가에 쓴 부품(src/의 common · pipeline · namer · illuminant · scoring · hint)을 그대로 불러 쓴다
#     → 이름 규칙 · 카메라 행렬 · p · 기준선 조명이 평가와 똑같다 ("이 앱의 정답률 = 평가 숫자"의 근거)
#   - ① 데이터셋 모드: Gehler 사진 (평가와 같은 처리, 다섯 방법 비교)
#   - ② 내 사진 모드: 폰 JPG/PNG (sRGB 감마 되돌리기 → Shades of Gray → sRGB 행렬 + D65→D50 순응 → Lab → 이름)
#   - ③ 실시간 카메라 모드: 폰 IP Webcam의 한 장(shot.jpg)을 ②와 똑같이 처리 (프레임마다 조명 다시 추정)
# 직접 실행하면(python app/engine.py) 평가 계산과 같은 이름이 나오는지 스스로 점검한다
import io
import json
import sys
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import colour
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import NEUTRALS, GEHLER_BLACK, GEHLER_SAT_RAW, gehler_camera, load_gehler_index, load_gehler_image  # noqa: E402
from pipeline import XYZ_STD, D50, load_color_matrices  # noqa: E402
from namer import name_color, lab_to_lch  # noqa: E402
from illuminant import load_methods, sog_many  # noqa: E402
from scoring import illum_xy, cct_duv  # noqa: E402
from hint import near_name  # noqa: E402

PARAMS = json.loads((ROOT / "reports" / "illum_params.json").read_text())
HINT = json.loads((ROOT / "reports" / "web_params.json").read_text())["hint"]
MATRICES = load_color_matrices()
ROI_R = 5                                   # 클릭한 점 둘레 반지름(픽셀) → 11×11 영역의 선형 평균
HALF = 0.5                                  # 영역의 절반 이상이 포화면 "너무 밝음"
DARK_RAW = 20                               # ① 블랙 레벨 뺀 가장 어두운 채널 평균 < 20 → 너무 어두움 (D3과 같은 기준)
JPG_CLIP, JPG_DARK = 250, 15                # ② 8비트 기준: 250 이상 = 포화, 가장 밝은 채널 평균 ≤ 15 = 너무 어두움
DISPLAY_MAX = 1100                          # 화면에 보낼 그림의 긴 변(픽셀)

# 화면 표시용 변환: XYZ(D50) → sRGB(D65) — Bradford 순응
_SRGB = colour.RGB_COLOURSPACES["sRGB"]
_D65 = _SRGB.whitepoint
_CAT_50_65 = colour.adaptation.matrix_chromatic_adaptation_VonKries(
    colour.xy_to_XYZ(D50), colour.xy_to_XYZ(_D65), transform="Bradford")
# ② 내 사진 모드: sRGB 선형 → XYZ(D65) → XYZ(D50). 이 순응을 빼면 회색이 모두 약간 푸르게 계산된다 (D13)
SRGB_TO_XYZ50 = np.linalg.inv(_CAT_50_65) @ _SRGB.matrix_RGB_to_XYZ


# ── 공통 ──────────────────────────────────────────────────────
def to_lab(xyz):
    return colour.XYZ_to_Lab(np.asarray(xyz, float), D50)


def lab_to_hex(lab):
    """Lab(D50) → 화면용 sRGB 색 견본 (#rrggbb)"""
    xyz = colour.Lab_to_XYZ(lab, D50)
    rgb = _SRGB.matrix_XYZ_to_RGB @ (_CAT_50_65 @ xyz)
    rgb = colour.cctf_encoding(np.clip(rgb, 0, 1), function="sRGB")
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


def gains(illum):
    """조명 색 → 이득 [r/g, 1, b/g] (pipeline.apply_illuminant와 같음)"""
    illum = np.asarray(illum, float)
    return illum / illum[1]


def describe(lab):
    """Lab → 이름 · 가까운 이름 · L/C/h"""
    L, C, h = lab_to_lch(lab)
    return {"name": name_color(lab), "near": near_name(lab, HINT),
            "lab": {"L": round(L, 1), "C": round(C, 1), "h": round(h, 1)}, "hex": lab_to_hex(lab)}


def cct_label(cct):
    if cct is None or not np.isfinite(cct):
        return "알 수 없음"
    return "따뜻한 조명" if cct < 4000 else ("주광" if cct <= 6500 else "푸른 조명")


def josa_ro(word):
    """이름에 붙일 (으)로: 받침 없음 · ㄹ받침 → 로 (보라로), 그 밖의 받침 → 으로 (주황으로 · 회색으로)"""
    c = ord(word[-1]) - 0xAC00
    if not 0 <= c <= 11171:
        return "(으)로"
    return "로" if c % 28 in (0, 8) else "으로"


def roi_box(x, y, w, h, r=ROI_R):
    x, y = int(round(x)), int(round(y))
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("클릭한 점이 사진 밖입니다")
    return max(0, x - r), max(0, y - r), min(w, x + r + 1), min(h, y + r + 1)


def to_png(rgb8):
    buf = io.BytesIO()
    Image.fromarray(rgb8).save(buf, format="PNG")
    return buf.getvalue()


def shrink(img, longest=DISPLAY_MAX):
    h, w = img.shape[:2]
    s = min(1.0, longest / max(h, w))
    return img if s == 1.0 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ── ① 데이터셋 모드 (Gehler, 평가와 같은 처리) ───────────────────────
@lru_cache(maxsize=1)
def _index():
    d = load_gehler_index(None)
    calib = set((ROOT / "splits" / "gehler_calib.txt").read_text().split())
    return d, {n: i for i, n in enumerate(d["names"])}, calib


def list_samples():
    d, _, calib = _index()
    return [{"name": n, "camera": c, "split": "보정용" if n in calib else "평가용"}
            for n, c in zip(d["names"], d["cams"])]


def exposure_scale(patches):
    """24칸 색 (24, 3) → 노출 배율 (pipeline.normalize_exposure와 같은 식: 회색 칸 G의 최소제곱)

    G 채널은 조명 이득으로 안 바뀌므로 모든 방법에 같은 배율이다 (D6)
    """
    nidx = [k - 1 for k in NEUTRALS]
    g, y = patches[nidx, 1], XYZ_STD[nidx, 1]
    return float(np.nansum(g * y) / np.nansum(g * g))


@lru_cache(maxsize=3)
def _gehler(name):
    """사진 하나를 읽고 조명 다섯 가지를 준비 (같은 사진을 여러 번 클릭할 때 다시 계산하지 않게 기억)"""
    d, pos, _ = _index()
    if name not in pos:
        raise KeyError(f"데이터셋에 없는 사진: {name}")
    k = pos[name]
    cam = gehler_camera(name)
    lin, valid = load_gehler_image(name)                 # 평가와 같은 입구: 포화 · 컬러체커 가림 · 블랙 레벨
    methods = load_methods("gehler")                     # {"gray_world": 1, "white_patch": inf, "shades_of_gray": p}
    est = sog_many(lin, valid, sorted(set(methods.values())))
    p = methods["shades_of_gray"]
    illums = {
        "기준선(고정)": np.array(PARAMS["baseline"][cam]),
        "Gray World": est[1],
        "White Patch": est[np.inf],
        f"Shades of Gray (p={p})": est[p],
        "상한선(정답 조명)": d["illum"][k],
    }
    return {"lin": lin.astype(np.float32), "cam": cam, "scale": exposure_scale(d["patches"][k]),
            "illums": illums, "main": f"Shades of Gray (p={p})", "base": "기준선(고정)", "gt": d["illum"][k]}


def name_camera_rgb(rgb, illum, cam, scale):
    """블랙 레벨 뺀 카메라 RGB (3,) + 조명 → Lab (평가 처리 ①~④: 이득 → 노출 → 행렬 → Lab)"""
    wb = np.asarray(rgb, float) / gains(illum) * scale
    return to_lab(MATRICES[cam] @ wb)


def illum_info(illum, cam):
    cct, _ = cct_duv(illum_xy(np.asarray(illum, float)[None], np.array([cam]), PARAMS["cct"]["how"]))
    c = float(cct[0])
    return {"rgb": [round(float(v), 4) for v in np.asarray(illum) / np.linalg.norm(illum)],
            "cct": int(round(c)) if np.isfinite(c) else None, "label": cct_label(c)}


def analyze_sample(name, x, y, norm=False):
    """norm=True면 x, y는 사진 너비 · 높이에 대한 비율(0~1) — 화면 그림 크기와 상관없이 같은 점"""
    g = _gehler(name)
    lin = g["lin"]
    h, w = lin.shape[:2]
    if norm:
        x, y = x * w, y * h
    x1, y1, x2, y2 = roi_box(x, y, w, h)
    roi = lin[y1:y2, x1:x2].reshape(-1, 3).astype(float)
    black = GEHLER_BLACK[g["cam"]]
    out = {"mode": "dataset", "roi": [x1, y1, x2, y2], "image_size": [w, h],
           "estimated_illuminant": illum_info(g["illums"][g["main"]], g["cam"])}
    if np.mean(np.any(roi + black > GEHLER_SAT_RAW, axis=1)) >= HALF:
        return {**out, "status": "uncertain", "reason": "too_bright", "message": "너무 밝아서(포화) 판단하기 어려워요"}
    mean = roi.mean(axis=0)
    if mean.min() < DARK_RAW:
        return {**out, "status": "uncertain", "reason": "too_dark", "message": "너무 어두워서 판단하기 어려워요"}
    rows = []
    for m, il in g["illums"].items():
        rows.append({"method": m, **describe(name_camera_rgb(mean, il, g["cam"], g["scale"]))})
    main = next(r for r in rows if r["method"] == g["main"])
    base = next(r for r in rows if r["method"] == g["base"])
    info = out["estimated_illuminant"]
    binfo = illum_info(g["illums"][g["base"]], g["cam"])        # 고정 보정이 가정하는 조명 (카메라별 보정용 평균)
    explain = None
    if base["name"] != main["name"] and info["cct"] and binfo["cct"]:
        # 기준선이 틀리는 이유 = 이 사진의 조명이 고정 보정이 가정한 조명과 달라서 → 어느 쪽으로 다른지 말해 준다
        side = "따뜻한" if info["cct"] < binfo["cct"] else "푸른"
        explain = (f"이 사진의 조명(약 {info['cct']:,}K)이 고정 보정이 가정한 조명(약 {binfo['cct']:,}K)보다 {side} 쪽이라, "
                   f"보정 전에는 '{base['name']}'{josa_ro(base['name'])} 보였어요")
    return {**out, "status": "ok", "reason": None, "color_name": main["name"], "near_name": main["near"],
            "baseline_name": base["name"], "baseline_illuminant": binfo, "explain": explain,
            "lab": main["lab"], "swatch": main["hex"], "methods": rows}


@lru_cache(maxsize=24)
def sample_image(name, view):
    """화면용 그림 (PNG). view = baseline(보정 전 = 고정 보정) · corrected(Shades of Gray로 보정)"""
    g = _gehler(name)
    il = g["illums"][g["base"] if view == "baseline" else g["main"]]
    img = shrink(g["lin"])
    wb = img.reshape(-1, 3).astype(float) / gains(il) * g["scale"]
    xyz = wb @ MATRICES[g["cam"]].T
    rgb = xyz @ (_SRGB.matrix_XYZ_to_RGB @ _CAT_50_65).T
    rgb = colour.cctf_encoding(np.clip(rgb, 0, 1), function="sRGB")
    return to_png((rgb.reshape(img.shape) * 255 + 0.5).astype(np.uint8))


# ── ② 내 사진 모드 (폰 JPG/PNG, 체험용) ────────────────────────────
def load_upload(data):
    """업로드 바이트 → 8비트 sRGB 배열 (세로 사진 회전 반영 · 긴 변 1,600 이하로 줄임)"""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    im.thumbnail((1600, 1600))
    return np.asarray(im, dtype=np.uint8).copy()


def upload_illum(srgb8):
    """사진 전체에서 Shades of Gray로 남은 색 기운 추정 (포화 픽셀은 뺌)"""
    lin = colour.cctf_decoding(srgb8.astype(float) / 255, function="sRGB")
    valid = ~np.any(srgb8 >= JPG_CLIP, axis=2)
    p = load_methods("gehler")["shades_of_gray"]
    return sog_many(lin, valid, [p])[p]


def cast_info(illum):
    """② 남은 색 기운 → 글자 (폰이 이미 화이트밸런스를 했으므로 K로 말하지 않음, D13)"""
    e = np.asarray(illum, float)
    ang = float(np.degrees(np.arccos(np.clip(e.sum() / (np.linalg.norm(e) * np.sqrt(3)), -1, 1))))
    if ang < 2:
        label = "거의 없음"
    else:
        label = "따뜻한 쪽(노랑 · 주황)" if e[0] > e[2] else "푸른 쪽"
    return {"rgb": [round(float(v), 4) for v in e / np.linalg.norm(e)], "cct": None, "label": f"남은 색 기운: {label}",
            "angle": round(ang, 1)}


def analyze_upload(srgb8, illum, x, y, norm=False):
    h, w = srgb8.shape[:2]
    if norm:
        x, y = x * w, y * h
    x1, y1, x2, y2 = roi_box(x, y, w, h)
    roi8 = srgb8[y1:y2, x1:x2].reshape(-1, 3)
    out = {"mode": "upload", "roi": [x1, y1, x2, y2], "image_size": [w, h], "estimated_illuminant": cast_info(illum)}
    if np.mean(np.any(roi8 >= JPG_CLIP, axis=1)) >= HALF:
        return {**out, "status": "uncertain", "reason": "too_bright", "message": "너무 밝아서(하얗게 날아가서) 판단하기 어려워요"}
    if roi8.astype(float).mean(axis=0).max() <= JPG_DARK:
        return {**out, "status": "uncertain", "reason": "too_dark", "message": "너무 어두워서 판단하기 어려워요"}
    lin = colour.cctf_decoding(roi8.astype(float) / 255, function="sRGB").mean(axis=0)   # 선형으로 바꾼 뒤 평균
    before = describe(to_lab(SRGB_TO_XYZ50 @ lin))                                       # 보정 전 = 사진 그대로
    after = describe(to_lab(SRGB_TO_XYZ50 @ (lin / gains(illum))))                       # Shades of Gray로 보정
    explain = None
    if before["name"] != after["name"]:
        explain = f"사진에 남은 색 기운 때문에 보정 전에는 '{before['name']}'{josa_ro(before['name'])} 보였어요"
    return {**out, "status": "ok", "reason": None, "color_name": after["name"], "near_name": after["near"],
            "baseline_name": before["name"], "explain": explain, "lab": after["lab"], "swatch": after["hex"],
            "methods": [{"method": "보정 전(사진 그대로)", **before}, {"method": "Shades of Gray로 보정", **after}]}


# ── ③ 실시간 카메라 모드 (IP Webcam, 체험용) ─────────────────────────
#   폰의 IP Webcam이 주는 한 장(shot.jpg)을 ② 내 사진 모드와 똑같이 처리한다 (계산 부품을 새로 만들지 않음)
LIVE_MAX = 640                              # 실시간은 빠르게: 긴 변 640으로 줄여서 조명 추정 · 화면 표시


def load_frame(data):
    """JPEG 바이트 → 8비트 sRGB 배열 (긴 변 LIVE_MAX 이하)"""
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((LIVE_MAX, LIVE_MAX))
    return np.asarray(im, dtype=np.uint8).copy()


def to_jpeg(rgb8, quality=80):
    buf = io.BytesIO()
    Image.fromarray(rgb8).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def analyze_live(srgb8, x, y):
    """프레임 한 장 + 비율 좌표(0~1) → ② 와 같은 결과 (mode만 live). 조명은 프레임마다 새로 추정"""
    res = analyze_upload(srgb8, upload_illum(srgb8), x, y, norm=True)
    return {**res, "mode": "live"}


# ── 직접 실행하면: 평가 계산과 같은지 점검 ──────────────────────────
if __name__ == "__main__":
    from pipeline import patches_to_lab, name_patches, normalize_exposure, apply_illuminant
    d, _, _ = _index()
    calib = (ROOT / "splits" / "gehler_calib.txt").read_text().split()[:5]
    n_same = n_all = 0
    for name in calib:
        g = _gehler(name)
        k = list(d["names"]).index(name)
        P = d["patches"][k]
        # 노출 배율이 pipeline과 같은지
        il = g["illums"][g["main"]]
        ref = normalize_exposure(apply_illuminant(P[None], il[None]))[0]
        mine = P / gains(il) * g["scale"]
        assert np.allclose(np.nan_to_num(ref), np.nan_to_num(mine)), "노출 배율이 평가와 다름"
        # 다섯 방법 모두 24칸 이름이 평가 계산(patches_to_lab → name_patches)과 같은지
        for m, il in g["illums"].items():
            want = name_patches(patches_to_lab(P[None], il[None], np.array([g["cam"]]), MATRICES))[0]
            for j in range(24):
                if np.isnan(P[j]).any():
                    continue
                got = describe(name_camera_rgb(P[j], il, g["cam"], g["scale"]))["name"]
                n_all += 1
                n_same += got == want[j]
    print(f"보정용 {len(calib)}장 × 다섯 방법 × 24칸: 평가 계산과 같은 이름 {n_same} / {n_all}")
    assert n_same == n_all
    # ② sRGB 회색은 D65 → D50 순응 뒤 채도 0이어야 함
    gray = describe(to_lab(SRGB_TO_XYZ50 @ np.array([0.2, 0.2, 0.2])))
    print(f"sRGB 회색 (0.2, 0.2, 0.2) → {gray['name']}, 채도 {gray['lab']['C']}")
    assert gray["lab"]["C"] < 0.5 and gray["name"] == "회색"
    print("조사:", ", ".join(f"{w}{josa_ro(w)}" for w in ["주황", "보라", "회색", "파랑", "노랑"]))
    assert [josa_ro(w) for w in ["주황", "보라", "회색"]] == ["으로", "로", "으로"]
    print("✅ engine.py 점검 통과 — 웹 계산이 평가 계산과 같음")