# src/08_fit_color_matrix.py
# 목적: 카메라 RGB → 표준 색(XYZ) 변환 행렬(3×3)을 "보정용 사진"으로만 맞춘다 (카메라별 1개씩)
#   원리: 정답 조명으로 화이트밸런스한 24칸 색  ≈  컬러체커 공식 표준 색(XYZ)  이 되도록
#         최소제곱(오차 제곱의 합이 가장 작게)으로 행렬 9개 값을 정한다
#   - 평가용 사진은 전혀 쓰지 않는다 (splits/gehler_calib.txt 만 사용)
#   - 정답표(색 이름)도 쓰지 않는다 → 정답표 확정 전에 해도 공정
#   - 결과: reports/color_matrix.json  (이후 모든 방법이 같은 행렬을 씀)
from pathlib import Path
import json
import numpy as np
import pandas as pd
import colour

ROOT = Path(__file__).resolve().parents[1]
SAT_RAW = 3300          # 원본 값이 이보다 크면 포화 의심 → 제외 (SFU가 정답 계산에 쓴 기준과 같음)
NEUTRALS = [20, 21, 22, 23]   # 밝기(노출) 맞추기에 쓸 회색 칸 (19번 흰색은 포화가 잦아 제외)

# ── 1. 컬러체커 공식 표준 색 (XYZ, D50 기준, 흰색 Y=1) ──────────
cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - Before November 2014"]
XYZ_STD = colour.xyY_to_XYZ(np.array(list(cc.data.values())))      # (24, 3)
D50 = cc.illuminant

# ── 2. 보정용 사진의 24칸 색 · 정답 조명 불러오기 ────────────────
index = pd.read_csv(ROOT / "outputs" / "gehler_index.csv").set_index("image")   # 02 결과
calib = (ROOT / "splits" / "gehler_calib.txt").read_text().split()
index = index.loc[calib]                                             # ⚠️ 보정용만 남김
print(f"보정용 사진 {len(index)}장 사용 (평가용은 열지 않음)")

cols = [f"p{k:02d}_{c}" for k in range(1, 25) for c in "rgb"]
P = index[cols].to_numpy().reshape(-1, 24, 3)        # (사진 수, 24, 3) 블랙 레벨 뺀 값, 빈칸 NaN
illum = index[["illum_r", "illum_g", "illum_b"]].to_numpy()        # (사진 수, 3)
black = index["black_level"].to_numpy()
cams = index["camera"].to_numpy()

# 포화 의심 칸 제외: 원본 값 = (뺀 값 + 블랙 레벨)
sat = ((P + black[:, None, None]) > SAT_RAW).any(axis=2)          # (사진 수, 24)
P[sat] = np.nan
print(f"포화 의심으로 제외한 칸: {sat.sum()}개")

# ── 3. 화이트밸런스 + 밝기 맞추기 ─────────────────────────────
# (1) 정답 조명으로 나누기. 조명은 G=1이 되게 맞춰서 R·B만 바뀌게 함 (색만 보정, 밝기는 그대로)
gains = illum / illum[:, [1]]                         # (사진 수, 3)  [r/g, 1, b/g]
WB = P / gains[:, None, :]
# (2) 사진마다 노출이 달라서, 회색 칸의 G 값이 표준 Y와 맞도록 사진별 배율 하나를 곱함
nidx = [k - 1 for k in NEUTRALS]
g = WB[:, nidx, 1]                                    # (사진 수, 4) 회색 칸 G
y = XYZ_STD[nidx, 1]                                  # (4,) 표준 Y
scale = np.nansum(g * y, axis=1) / np.nansum(g * g, axis=1)   # 사진별 최소제곱 배율
WB = WB * scale[:, None, None]

# ── 4. 카메라별로 3×3 행렬 맞추기 ──────────────────────────────
result = {}
for cam in ["1D", "5D"]:
    sel = cams == cam
    A = WB[sel].reshape(-1, 3)                        # 입력: 카메라 RGB (칸 수, 3)
    B = np.tile(XYZ_STD, (sel.sum(), 1))              # 목표: 표준 XYZ  (칸 수, 3)
    ok = ~np.isnan(A).any(axis=1)                     # 빈칸·포화 칸 제외
    Mt, *_ = np.linalg.lstsq(A[ok], B[ok], rcond=None)   # A @ Mt ≈ B 가 되는 Mt (3×3)
    M = Mt.T                                          # XYZ = M @ rgb 형태로 저장

    # 맞춘 정도 확인: 보정용 사진에서 표준 색과의 색 차이(ΔE2000)
    lab_fit = colour.XYZ_to_Lab(A[ok] @ Mt, D50)
    lab_std = colour.XYZ_to_Lab(B[ok], D50)
    dE = colour.delta_E(lab_fit, lab_std, method="CIE 2000")
    patch_id = np.tile(np.arange(1, 25), sel.sum())[ok]
    per_patch = pd.Series(dE).groupby(patch_id).median()

    print(f"\n[{cam}] 사진 {sel.sum()}장, 사용한 칸 {ok.sum()}개 (정할 값 9개)")
    print(f"  ΔE2000 중앙값 {np.median(dE):.2f}, 평균 {dE.mean():.2f}, 90% {np.percentile(dE, 90):.2f}")
    print(f"  칸별 ΔE 중앙값이 큰 칸: " +
          ", ".join(f"{k}번 {v:.1f}" for k, v in per_patch.sort_values(ascending=False).head(4).items()))
    print("  M =\n" + np.array2string(M, precision=4, suppress_small=True))

    result[cam] = {"M": M.round(6).tolist(), "n_images": int(sel.sum()), "n_patches": int(ok.sum()),
                   "dE2000_median": round(float(np.median(dE)), 3),
                   "dE2000_mean": round(float(dE.mean()), 3)}

result["note"] = ("XYZ(D50, 흰색 Y=1) = M @ (블랙 레벨 뺀 RGB / [r/g, 1, b/g] 조명 이득 × 노출 배율). "
                  "보정용 사진(splits/gehler_calib.txt)으로만 맞춤")
(ROOT / "reports" / "color_matrix.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print("\n저장: reports/color_matrix.json")