# src/10_check_camera_white.py
# 목적: 중간 점검 Y4 · Y5 · Y6 확인 (보정용 사진과 데이터 값만 사용, 색 이름·채점은 하지 않음)
#   Y4  Canon 1D의 색 변환 오차(ΔE 5.03)가 5D(2.55)의 2배 → 1D만 채널 순서가 다른 건 아닌지
#   Y5  흰색(19번) 칸이 9장에서 빈칸 → "어두워서"가 아니라 "너무 밝아서(포화)" 뺀 것인지
#   Y6  SFU판 정답 조명의 알려진 문제(Hemrit et al. 2018)가 우리 사진 몇 장에 해당하는지
import json
import numpy as np
import pandas as pd
import colour
from common import (ROOT, GEHLER_DIR, GEHLER_BLACK, GEHLER_SAT_RAW,
                    read_rgb, angular_error, load_gehler_index)
from pipeline import XYZ_STD, D50, apply_illuminant, normalize_exposure, fit_matrix, to_xyz, to_lab

d = load_gehler_index("calib")                     # 보정용만 (평가용은 열지 않음)
wb = normalize_exposure(apply_illuminant(d["patches"], d["illum"]))
LAB_STD = colour.XYZ_to_Lab(XYZ_STD, D50)

# ════════════════════════════════════════════════════════════
# Y4-①  "24칸 값에서 R·B를 바꿔 행렬을 다시 맞추기"는 확인이 되지 않는다
#   이유: 24칸 값과 정답 조명이 "같이" 바뀌면, 3×3 행렬이 그 바뀜을 그대로 흡수한다
#         (행렬의 1열과 3열이 서로 자리만 바꿀 뿐) → ΔE가 똑같이 나온다
# ════════════════════════════════════════════════════════════
sel = d["cams"] == "1D"
A = wb[sel].reshape(-1, 3)
B = np.tile(XYZ_STD, (sel.sum(), 1))
M, ok = fit_matrix(A, B)
M_sw, _ = fit_matrix(A[:, ::-1], B)                # R·B 바꾼 입력으로 다시 맞추기
dE = lambda Mx, X: np.median(colour.delta_E(colour.XYZ_to_Lab(X[ok] @ Mx.T, D50),
                                            colour.XYZ_to_Lab(B[ok], D50), method="CIE 2000"))
print("[Y4-①] 1D 24칸 값 R·B 바꿔서 행렬 다시 맞추기")
print(f"  원래 ΔE 중앙값 {dE(M, A):.3f}  /  R·B 바꿈 {dE(M_sw, A[:, ::-1]):.3f}")
print("  → 같으면: 행렬이 순서 차이를 흡수한 것. 이 방법으로는 채널 순서를 알 수 없음\n")

# ════════════════════════════════════════════════════════════
# Y4-②  진짜 확인: "사진 픽셀"의 순서가 "정답 파일(.mat)"의 순서와 같은가?
#   순서가 어긋날 수 있는 곳은 여기뿐이다 (사진은 OpenCV로, 정답은 .mat에서 읽음)
#   방법: Gray World(사진 평균 색 = 조명 색)로 조명을 추정해 정답 조명과 비교
#         read_rgb 순서가 맞으면 오차가 작고, R·B를 일부러 뒤집으면 커져야 한다
#   → 카메라별로 따로 본다. 1D만 반대로 나오면 1D의 채널 순서가 다른 것
# ════════════════════════════════════════════════════════════
print("[Y4-②] 사진 픽셀 순서 vs 정답 파일 순서 (Gray World 각도 오차 중앙값)")
for cam in ["1D", "5D"]:
    idx = np.where(d["cams"] == cam)[0]
    e_rgb, e_bgr = [], []
    for i in idx:
        raw = read_rgb(GEHLER_DIR / "png" / d["names"][i]).astype(np.float64)
        keep = ~np.any(raw > GEHLER_SAT_RAW, axis=2)            # 포화 픽셀 제외
        lin = np.clip(raw - GEHLER_BLACK[cam], 0, None)
        est = lin[keep].mean(axis=0)                            # Gray World 추정 (R, G, B)
        e_rgb.append(angular_error(est, d["illum"][i]))
        e_bgr.append(angular_error(est[::-1], d["illum"][i]))
    m_rgb, m_bgr = np.median(e_rgb), np.median(e_bgr)
    verdict = "✅ 순서 맞음" if m_rgb < m_bgr else "⚠️ 순서가 반대일 수 있음"
    print(f"  [{cam}] {len(idx):3d}장  read_rgb 순서 {m_rgb:5.2f}도  /  R·B 뒤집음 {m_bgr:5.2f}도  {verdict}")
print()

# ════════════════════════════════════════════════════════════
# Y4-③  순서 문제가 아니라면: 1D 오차가 "골고루" 큰가, "몇 장" 때문인가?
#   골고루 크면 → 카메라(센서) 특성 / 몇 장만 크면 → 그 사진들의 문제(노출 등)
# ════════════════════════════════════════════════════════════
mats = {cam: np.array(v["M"]) for cam, v in
        json.loads((ROOT / "reports" / "color_matrix.json").read_text()).items() if cam != "note"}
lab = to_lab(to_xyz(wb, d["cams"], mats))                       # (n, 24, 3)
de_img = colour.delta_E(lab, np.broadcast_to(LAB_STD, lab.shape), method="CIE 2000")  # (n, 24)
per_img = np.nanmedian(de_img, axis=1)                          # 사진별 ΔE 중앙값
print("[Y4-③] 사진별 ΔE 중앙값의 분포")
for cam in ["1D", "5D"]:
    v = per_img[d["cams"] == cam]
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    print(f"  [{cam}] {len(v):3d}장  하위 25% {q1:4.2f} · 중앙 {q2:4.2f} · 상위 25% {q3:4.2f} · 최대 {v.max():4.2f}")
worst = np.argsort(per_img)[::-1][:5]
print("  ΔE가 가장 큰 사진 5장: " + ", ".join(f"{d['names'][i]}({d['cams'][i]}) {per_img[i]:.1f}" for i in worst))
print("  → 1D의 '하위 25%'도 5D의 중앙보다 크면 골고루 큰 것 (사진 몇 장 탓이 아님)\n")

# ════════════════════════════════════════════════════════════
# Y5  흰색(19번)이 빈칸인 사진: 20번(밝은 회색) 값 확인
#   논리: 흰색은 20번보다 항상 밝다. 20번이 멀쩡히 남아 있는데 19번만 비었다면
#         "어두워서"(가장 어두운 채널 < 20)는 불가능 → 너무 밝아서 뺀 것
#   ※ 03에서 이미 568장 전체의 빈칸 위치를 본 것과 같은 "데이터 확인"이라 전체를 본다
#     (색 이름·조명 추정 결과는 보지 않으므로 평가용 봉인과 무관)
# ════════════════════════════════════════════════════════════
#   흰색이 얼마나 밝았을지 추정: 남아 있는 회색 칸 중 가장 밝은 칸(20 → 21 → 22 → 23 순서) 값
#   × 표준 밝기 Y 비율 (예: 20번이면 흰색 0.90 ÷ 20번 0.59 ≈ 1.55배)
#   20번까지 비었으면 21번으로 추정한다 → 회색 칸이 남아 있다는 것 자체가 "어둡지 않다"는 증거
idx_csv = pd.read_csv(ROOT / "outputs" / "gehler_index.csv")    # 포화 처리 전, .mat 빈칸 그대로
miss19 = idx_csv[idx_csv["p19_r"].isna()]
print(f"[Y5] 19번(흰색)이 빈칸인 사진: {len(miss19)}장  (포화 기준: 원본 값 {GEHLER_SAT_RAW})")
n_sat = 0
for _, r in miss19.iterrows():
    why, est19, ref = "회색 칸 20~23번이 모두 빈칸 → 판단 불가", np.full(3, np.nan), "-"
    for k in [20, 21, 22, 23]:                                   # 밝은 회색부터
        pk = np.array([r[f"p{k}_r"], r[f"p{k}_g"], r[f"p{k}_b"]])  # 블랙 레벨 뺀 값
        if np.isnan(pk).any():
            continue                                             # 이 칸도 비었으면 다음 칸으로
        ratio = XYZ_STD[18, 1] / XYZ_STD[k - 1, 1]               # 흰색 Y ÷ 이 칸 Y
        est19 = pk * ratio + r["black_level"]                    # 흰색의 원본 값 추정
        ref = f"{k}번×{ratio:.2f}"
        if est19.max() >= GEHLER_SAT_RAW:
            why = "포화 기준 이상 → 너무 밝아서 뺀 것"
            n_sat += 1
        else:
            why = "포화 기준 미만 → 원인 따로 확인"
        break
    print(f"  {r['image']:14s} {r['camera']}  흰색 추정({ref:9s}) R,G,B = "
          f"{', '.join(f'{x:5.0f}' for x in est19)}  → {why}")
print(f"  → {n_sat}/{len(miss19)}장이 '너무 밝아서'. 모두 그렇다면 데이터 노트의 빈칸 설명을 "
      f"'어둡거나 포화된 칸'으로 고친다")

# ════════════════════════════════════════════════════════════
# Y6  SFU판 정답 조명의 알려진 문제가 우리 사진 몇 장에 해당하나?
#   Hemrit et al. 2018(REC 정답)이 지적한 것: SFU 정답은 R·G·B를 "채널마다 따로" 가장 큰 회색 칸에서
#   가져와서, 몇 장은 세 채널이 서로 다른 칸에서 왔다 (같은 칸을 써야 맞음)
#   우리 02의 검산도 "19~24번 채널별 최댓값"이라 같은 방식 → 우리 데이터에서 직접 세어 본다
#   ① 세 채널의 최댓값이 서로 다른 칸에서 온 사진
#   ② 정답에 쓰인 칸의 원본 값이 포화 기준(3300)을 넘는 사진
#   ※ Y5처럼 데이터 값만 보는 확인이라 568장 전체를 본다 (조명 추정 · 채점 결과는 보지 않음)
# ════════════════════════════════════════════════════════════
neu_cols = [[f"p{k}_{c}" for c in "rgb"] for k in range(19, 25)]
vals = np.stack([idx_csv[c].to_numpy() for c in neu_cols], axis=1)   # (568, 6칸, 3채널)
src = np.nanargmax(np.where(np.isnan(vals), -np.inf, vals), axis=1)  # 채널마다 최댓값이 나온 칸 (0=19번)
gt_raw = np.nanmax(vals, axis=1) + idx_csv["black_level"].to_numpy()[:, None]
mixed = ~((src[:, 0] == src[:, 1]) & (src[:, 1] == src[:, 2]))      # ① 세 채널의 칸이 다름
over = (gt_raw > GEHLER_SAT_RAW).any(axis=1)                           # ② 정답 칸이 포화 의심
print(f"\n[Y6] SFU 정답 조명 점검 (568장 전체)")
print(f"  ① 세 채널이 서로 다른 칸에서 온 사진: {mixed.sum()}장")
for i in np.where(mixed)[0]:
    print(f"     {idx_csv['image'][i]:14s} {idx_csv['camera'][i]}  R·G·B 칸 = "
          f"{', '.join(str(k + 19) for k in src[i])}번")
print(f"  ② 정답에 쓰인 칸의 원본 값이 {GEHLER_SAT_RAW} 초과인 사진: {over.sum()}장")
for i in np.where(over)[0]:
    print(f"     {idx_csv['image'][i]:14s} {idx_csv['camera'][i]}  원본 R,G,B = "
          f"{', '.join(f'{x:5.0f}' for x in gt_raw[i])}")
print("  → 해당 사진 수를 데이터 노트 한계에 기록. 몇 장뿐이면 상한선에 주는 영향은 작다")