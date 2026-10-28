# src/09_pilot_chart_detect.py
# 목적: Gehler 사진 속 컬러체커 위치를 자동으로 찾는 방법 2가지를 "보정용 30장"으로 시험해
#       실제 실패율을 잰다 (결정 기록 D8의 다음 행동)
#   방법 A: colour-checker-detection 패키지 (모양으로 찾기) + 찾은 24칸 색을 데이터셋 값과 비교해 검증
#   방법 B: 정답 색으로 찾기 — 이 사진의 24칸 실제 색(.mat)을 알고 있으므로,
#           그 색들이 한곳에 모여 있는 자리를 찾는다
#   결과: outputs/chart_pilot/*.jpg (사진마다 A=빨강, B=초록 상자), 한눈에 보기 그림, 요약 표
#   ⚠️ 위치를 "찾을 때만" 정답 조명으로 보기 좋게 바꾼다. 조명 추정에는 원본 선형 사진을 쓴다
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import colour
from colour_checker_detection import detect_colour_checkers_segmentation
from common import ROOT, GEHLER_DIR, GEHLER_BLACK, read_rgb, load_gehler_index

N_PILOT = 30
SEED = 0
OUT = ROOT / "outputs" / "chart_pilot"
OUT.mkdir(parents=True, exist_ok=True)


def to_display(lin, illum, scale):
    """선형 값 → 정답 조명으로 색 맞춤 → 밝기 맞춤 → sRGB 감마 (사람 눈·검출기용 보기 좋은 그림)"""
    gains = np.asarray(illum, float) / illum[1]
    v = np.clip(lin / gains * scale, 0, 1)
    return colour.cctf_encoding(v)                      # 선형 → sRGB 감마


# ── 방법 A: colour-checker-detection ───────────────────────────
def detect_a(disp, expected_disp):
    """찾은 사각형(원본 좌표 4점)과 24칸 색 일치 정도(작을수록 좋음)를 돌려준다. 못 찾으면 (None, inf)"""
    try:
        results = detect_colour_checkers_segmentation(disp, additional_data=True)
    except Exception:
        return None, np.inf
    best_quad, best_err = None, np.inf
    s = disp.shape[1] / 1440                              # 라이브러리는 가로 1440으로 줄여서 계산함
    ok = ~np.isnan(expected_disp).any(axis=1)
    for r in results:
        sw = np.asarray(r.swatch_colours)
        if sw.shape != (24, 3):
            continue
        # 컬러체커가 뒤집혀 찍혔을 수 있으니 정방향·역방향 중 더 잘 맞는 쪽
        err = min(np.median(np.abs(sw[ok] - expected_disp[ok])),
                  np.median(np.abs(sw[::-1][ok] - expected_disp[ok])))
        if err < best_err:
            best_quad, best_err = np.asarray(r.quadrilateral) * s, err
    return best_quad, best_err


# ── 방법 B: 정답 색으로 찾기 ───────────────────────────────────
def detect_b(lin, patches, tol=0.2, floor=20.0):
    """24칸 색과 비슷한 픽셀들이 가장 촘촘히 모인 자리 → 상자(x1, y1, x2, y2)와 찾은 칸 비율"""
    f = 2                                                 # 계산을 빠르게 하려고 절반 크기에서 찾음
    small = cv2.resize(lin.astype(np.float32), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    ok = np.where(~np.isnan(patches).any(axis=1))[0]      # 빈칸이 아닌 칸만
    masks = []
    for k in ok:
        c = patches[k].astype(np.float32)
        # 채널마다 "몇 배 차이 나는지"(로그 비율)로 비교 → 밝기 차이에 덜 민감
        d = np.max(np.abs(np.log((small + floor) / (c + floor))), axis=2)
        masks.append((d < tol).astype(np.float32))
    masks = np.stack(masks)                               # (칸 수, H, W)

    best = None
    for w in [16, 24, 36, 54, 80, 120, 180, 270]:        # 컬러체커 크기를 모르니 여러 창 크기로
        present = np.stack([cv2.blur(m, (w, w)) > 0 for m in masks])   # 창 안에 그 색이 있나
        count = present.sum(axis=0)                       # 창 안에 있는 서로 다른 칸 수
        y, x = np.unravel_index(np.argmax(count), count.shape)
        cand = (count[y, x], w, x, y)
        if best is None or cand[0] > best[0]:
            best = cand
        if count[y, x] >= 0.85 * len(ok):                 # 충분히 모였으면 가장 작은 창에서 멈춤
            best = cand
            break
    n_found, w, cx, cy = best
    # 창 주변에서 칸별로 맞는 픽셀들의 가운데 위치 → 그 점들을 감싸는 상자 + 여유
    h, wd = small.shape[:2]
    r = int(w * 0.75)
    y1, y2, x1, x2 = max(cy - r, 0), min(cy + r, h), max(cx - r, 0), min(cx + r, wd)
    pts = []
    for m in masks:
        ys, xs = np.nonzero(m[y1:y2, x1:x2])
        if len(ys):
            pts.append((np.median(xs) + x1, np.median(ys) + y1))
    pts = np.array(pts)
    pad = 0.4 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 4)   # 넉넉하게: 컬러체커가 삐져나오지 않게
    box = np.array([pts[:, 0].min() - pad, pts[:, 1].min() - pad,
                    pts[:, 0].max() + pad, pts[:, 1].max() + pad]) * f
    return box, n_found / len(ok)


# ── 시험할 사진 30장 고르기 (보정용에서만, 카메라 비율대로) ───────────
d = load_gehler_index("calib")
rng = np.random.default_rng(SEED)
idx_1d = rng.choice(np.where(d["cams"] == "1D")[0], 5, replace=False)
idx_5d = rng.choice(np.where(d["cams"] == "5D")[0], N_PILOT - 5, replace=False)
picks = sorted(np.concatenate([idx_1d, idx_5d]))

rows, thumbs = [], []
for n, i in enumerate(picks, 1):
    name, cam, illum = d["names"][i], d["cams"][i], d["illum"][i]
    raw = read_rgb(GEHLER_DIR / "png" / name).astype(np.float64)       # ⚠️ RGB 순서로 읽기
    lin = np.clip(raw - GEHLER_BLACK[cam], 0, None)
    patches = d["patches"][i]                                            # 이 사진의 24칸 실제 색

    gains = illum / illum[1]
    scale = 0.9 / np.percentile(lin / gains, 99)                        # 밝은 쪽 1%가 0.9 근처가 되게
    disp = to_display(lin, illum, scale)
    expected = to_display(patches, illum, scale)                         # 24칸 색도 같은 방식으로

    quad, err_a = detect_a(disp, expected)
    box_b, found_b = detect_b(lin, patches)
    pass_a = quad is not None and err_a < 0.08
    pass_b = found_b >= 0.85

    # 확인용 그림: A = 빨강 사각형, B = 초록 상자
    vis = (disp[:, :, ::-1] * 255).astype(np.uint8).copy()              # 저장용 BGR
    if quad is not None:
        cv2.polylines(vis, [quad.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 6)
    x1, y1, x2, y2 = box_b.astype(int)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 6)
    label = f"{n:02d} {name}  A:{'O' if pass_a else 'X'}  B:{found_b:.0%}"
    cv2.putText(vis, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
    cv2.imwrite(str(OUT / f"{n:02d}_{Path(name).stem}.jpg"), cv2.resize(vis, None, fx=0.5, fy=0.5))
    thumbs.append(cv2.resize(vis, (360, 240)))

    rows.append({"no": n, "image": name, "camera": cam,
                 "A_찾음": quad is not None, "A_색오차": round(float(err_a), 3), "A_통과": pass_a,
                 "B_찾은칸비율": round(float(found_b), 2), "B_통과": pass_b})
    print(f"[{n:02d}/{len(picks)}] {name:14s} A {'O' if pass_a else 'X'} (오차 {err_a:.3f})  "
          f"B {'O' if pass_b else 'X'} ({found_b:.0%})")

# ── 요약 ─────────────────────────────────────────────────────
res = pd.DataFrame(rows)
res.to_csv(OUT / "pilot_summary.csv", index=False)
print(f"\n자동 판정 통과율 — A: {res['A_통과'].mean():.0%}  B: {res['B_통과'].mean():.0%}  "
      f"둘 중 하나라도: {(res['A_통과'] | res['B_통과']).mean():.0%}")
print("⚠️ 자동 판정은 참고용. 최종 실패율은 한눈에 보기 그림을 '눈으로' 확인해서 정한다")

# 30장 한눈에 보기 (6열 × 5행)
grid = [np.hstack(thumbs[r * 6:(r + 1) * 6]) for r in range(len(thumbs) // 6)]
cv2.imwrite(str(OUT / "_contact_sheet.jpg"), np.vstack(grid))
print(f"저장: {OUT.relative_to(ROOT)}/_contact_sheet.jpg  (빨강 = A, 초록 = B)")