# src/08_fit_color_matrix.py
# 목적: 카메라 RGB → 표준 색(XYZ) 변환 행렬(3×3)을 "보정용 사진"으로만 맞춘다 (카메라별 1개씩)
#   원리: 정답 조명으로 화이트밸런스한 24칸 색  ≈  컬러체커 공식 표준 색(XYZ)  이 되도록
#         최소제곱(오차 제곱의 합이 가장 작게)으로 행렬 9개 값을 정한다
#   - 평가용 사진은 전혀 쓰지 않는다 (보정용만 불러옴)
#   - 정답표(색 이름)도 쓰지 않는다 → 정답표 확정 전에 해도 공정
#   - 처리 단계는 pipeline.py의 함수를 그대로 사용 → 이후 채점과 똑같은 처리가 보장됨
#   - 결과: reports/color_matrix.json
import json
import numpy as np
import pandas as pd
import colour
from common import ROOT, load_gehler_index
from pipeline import XYZ_STD, D50, apply_illuminant, normalize_exposure, fit_matrix

# ── 1. 보정용 사진만 불러오기 (빈칸·포화 의심 칸은 NaN) ──────────
d = load_gehler_index("calib")
print(f"보정용 사진 {len(d['names'])}장 사용 (평가용은 열지 않음)")
print(f"포화 의심으로 제외한 칸: {d['n_saturated']}개")

# ── 2. 정답 조명으로 화이트밸런스 + 노출 맞추기 (pipeline ①·②) ──
wb = normalize_exposure(apply_illuminant(d["patches"], d["illum"]))

# ── 3. 카메라별로 3×3 행렬 맞추기 ──────────────────────────────
result = {}
for cam in ["1D", "5D"]:
    sel = d["cams"] == cam
    A = wb[sel].reshape(-1, 3)                        # 입력: 카메라 RGB (칸 수, 3)
    B = np.tile(XYZ_STD, (sel.sum(), 1))              # 목표: 표준 XYZ  (칸 수, 3)
    M, ok = fit_matrix(A, B)

    # 맞춘 정도 확인: 보정용 사진에서 표준 색과의 색 차이(ΔE2000)
    lab_fit = colour.XYZ_to_Lab(A[ok] @ M.T, D50)
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