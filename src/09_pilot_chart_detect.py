# src/09_pilot_chart_detect.py  (3차 시험)
# 목적: Gehler 사진 속 컬러체커 위치를 "정답 색으로 찾기"로 자동 검출하고, 보정용 30장으로 시험한다
#   원리: 이 사진의 24칸 실제 색(.mat)을 알고 있으므로, 그 색들이 한곳에 모여 있는 자리를 찾는다
#   1차 시험(커밋 기록 참고) 결과: 모양으로 찾기 12/30, 정답 색으로 찾기 17/30 (눈으로 확인)
#   2차에서 고친 것
#     ① 회색 6칸·피부색 2칸은 사진 어디에나 흔하므로, "모였는지" 셀 때는 유채색 16칸(3~18번)만 센다
#     ② "충분히 모이면 멈추기"를 없애고, 모든 창 크기 중 가장 많이 모인 곳을 고른다 (같으면 작은 창)
#     ③ 크게 찍힌 컬러체커도 찾도록 창 크기를 원본 기준 1,080픽셀까지 늘린다
#   2차 시험 결과: 22/30 정확, 5장 일부만, 2장 엉뚱한 곳 (눈으로 확인) — 큰 컬러체커의 일부만 잡힘
#   3차에서 고친 것
#     ④ 창 안에 그 색 픽셀이 "하나라도"가 아니라 "칸 넓이의 1/4 이상" 있어야 그 색이 있다고 본다
#        → 큰 컬러체커의 일부만 담은 작은 창, 흩어진 픽셀 몇 개로 생기는 가짜 자리를 걸러 냄
#   확인 시험: ROUND="check"면 1~3차에 안 쓴 보정용 30장으로 3차 방식을 그대로 돌려 진짜 성공률을 잰다
#   결과: outputs/chart_pilot/ (ROUND="tune") 또는 outputs/chart_pilot_check/ (ROUND="check")
#   ⚠️ 자동 판정은 스스로 매긴 점수라 참고용. 최종 판단은 그림을 "눈으로" 확인해서 한다
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import colour
from common import ROOT, GEHLER_DIR, GEHLER_BLACK, read_rgb, load_gehler_index

N_PILOT = 30
SEED = 0                                   # 1차와 같은 30장을 뽑아 비교
OUT = ROOT / "outputs" / "chart_pilot"
OUT.mkdir(parents=True, exist_ok=True)
CHROMATIC = list(range(2, 18))             # 3~18번 칸 (0부터 세면 2~17)
WINDOWS = [16, 24, 36, 54, 80, 120, 180, 270, 400, 540]   # 절반 크기 기준 창 크기 (원본은 2배)


def to_display(lin, illum, scale):
    """선형 값 → 정답 조명으로 색 맞춤 → 밝기 맞춤 → sRGB 감마 (사람 눈 확인용 그림)"""
    gains = np.asarray(illum, float) / illum[1]
    return colour.cctf_encoding(np.clip(lin / gains * scale, 0, 1))


def detect_chart(lin, patches, tol=0.2, floor=20.0):
    """유채색 16칸과 비슷한 픽셀들이 가장 많이 모인 자리 → 상자(x1, y1, x2, y2), 찾은 칸 비율, 창 크기"""
    f = 2                                                 # 계산을 빠르게 하려고 절반 크기에서 찾음
    small = cv2.resize(lin.astype(np.float32), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    ks = [k for k in CHROMATIC if not np.isnan(patches[k]).any()]   # 빈칸이 아닌 유채색 칸만
    masks = []
    for k in ks:
        c = patches[k].astype(np.float32)
        # 채널마다 "몇 배 차이 나는지"(로그 비율)로 비교 → 밝기 차이에 덜 민감
        d = np.max(np.abs(np.log((small + floor) / (c + floor))), axis=2)
        masks.append((d < tol).astype(np.float32))

    best = (-1, None, None, None)                         # (모인 칸 수, 창, x, y)
    for w in WINDOWS:                                     # 작은 창부터 → 같은 수면 작은 창이 남음
        # 창이 컬러체커만 하다면 칸 하나는 창 폭의 약 1/7 → 그 넓이의 1/4 이상 있어야 "그 색이 있다"
        min_px = max(2.0, 0.25 * (w / 7) ** 2)
        count = sum((cv2.blur(m, (w, w)) * w * w >= min_px).astype(np.int32) for m in masks)
        y, x = np.unravel_index(np.argmax(count), count.shape)
        if count[y, x] > best[0]:                         # "더 많이" 모였을 때만 바꿈
            best = (count[y, x], w, x, y)
    n_found, w, cx, cy = best

    # 창 주변에서 칸별로 맞는 픽셀들의 가운데 위치 → 그 점들을 감싸는 상자 + 넉넉한 여유
    h, wd = small.shape[:2]
    r = int(w * 0.75)
    y1, y2, x1, x2 = max(cy - r, 0), min(cy + r, h), max(cx - r, 0), min(cx + r, wd)
    pts = []
    for m in masks:
        ys, xs = np.nonzero(m[y1:y2, x1:x2])
        if len(ys):
            pts.append((np.median(xs) + x1, np.median(ys) + y1))
    pts = np.array(pts)
    pad = 0.4 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 4)   # 컬러체커가 삐져나오지 않게
    box = np.array([pts[:, 0].min() - pad, pts[:, 1].min() - pad,
                    pts[:, 0].max() + pad, pts[:, 1].max() + pad]) * f
    return box, n_found / len(ks), w * f


# ── 시험할 사진 30장 ─────────────────────────────────────────
# ROUND = "tune"  : 1~3차에 쓴 30장 (고칠 때 본 사진 → 검출기의 "보정용")
# ROUND = "check" : 위 30장을 뺀 나머지 보정용 사진에서 새로 30장 (처음 보는 사진 → 검출기의 "진짜 실력")
ROUND = "check"
d = load_gehler_index("calib")
rng = np.random.default_rng(SEED)
idx_1d = rng.choice(np.where(d["cams"] == "1D")[0], 5, replace=False)
idx_5d = rng.choice(np.where(d["cams"] == "5D")[0], N_PILOT - 5, replace=False)
picks = sorted(np.concatenate([idx_1d, idx_5d]))          # 1~3차의 30장 (seed 0)
if ROUND == "check":
    rest = np.setdiff1d(np.arange(len(d["names"])), picks)   # 1~3차 30장 빼기
    rest_1d = [i for i in rest if d["cams"][i] == "1D"]
    rest_5d = [i for i in rest if d["cams"][i] == "5D"]
    rng2 = np.random.default_rng(1)                           # 다른 seed로 새로 뽑기
    n_1d = min(5, len(rest_1d))                               # 남은 사진이 모자라면 있는 만큼만
    n_5d = min(N_PILOT - n_1d, len(rest_5d))
    picks = sorted(np.concatenate([rng2.choice(rest_1d, n_1d, replace=False),
                                   rng2.choice(rest_5d, n_5d, replace=False)]))
    OUT = ROOT / "outputs" / "chart_pilot_check"              # 1~3차 결과를 덮어쓰지 않게
    OUT.mkdir(parents=True, exist_ok=True)
print(f"ROUND = {ROUND}: 보정용 사진 {len(picks)}장으로 시험 → {OUT.relative_to(ROOT)}/")

rows, thumbs = [], []
for n, i in enumerate(picks, 1):
    name, cam, illum = d["names"][i], d["cams"][i], d["illum"][i]
    raw = read_rgb(GEHLER_DIR / "png" / name).astype(np.float64)       # ⚠️ RGB 순서로 읽기
    lin = np.clip(raw - GEHLER_BLACK[cam], 0, None)
    box, found, win = detect_chart(lin, d["patches"][i])

    gains = illum / illum[1]
    scale = 0.9 / np.percentile(lin / gains, 99)
    vis = (to_display(lin, illum, scale)[:, :, ::-1] * 255).astype(np.uint8).copy()   # 저장용 BGR
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 6)
    cv2.putText(vis, f"{n:02d} {name}  {found:.0%}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
    cv2.imwrite(str(OUT / f"{n:02d}_{Path(name).stem}.jpg"), cv2.resize(vis, None, fx=0.5, fy=0.5))
    thumbs.append(cv2.resize(vis, (360, 240)))

    rows.append({"no": n, "image": name, "camera": cam, "찾은칸비율": round(float(found), 2),
                 "창크기": win, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    print(f"[{n:02d}/{len(picks)}] {name:14s} 유채색 칸 {found:.0%} 모임 (창 {win}px)")

pd.DataFrame(rows).to_csv(OUT / "pilot_summary.csv", index=False)
grid = [np.hstack(thumbs[r * 6:(r + 1) * 6]) for r in range(len(thumbs) // 6)]
cv2.imwrite(str(OUT / "_contact_sheet.jpg"), np.vstack(grid))
print(f"\n저장: {OUT.relative_to(ROOT)}/_contact_sheet.jpg  — 초록 상자가 컬러체커를 덮는지 눈으로 확인")