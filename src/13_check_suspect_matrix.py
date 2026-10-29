# src/13_check_suspect_matrix.py
# 목적: (점검) 정답 조명이 의심스러운 보정용 사진 8D5U5541이 1D 색 변환 행렬에 얼마나 영향을 줬는지 숫자로 남긴다 (D9 보완)
#   - D9 결정: 보정 단계는 그대로 둔다. 이 스크립트는 "영향이 이 정도였다"는 근거만 만든다
#   - 방법: 1D 보정용 29장으로 맞춘 행렬(지금 쓰는 것) vs 8D5U5541을 뺀 28장으로 맞춘 행렬
#           → 같은 28장 · 같은 칸에서 색 차이(ΔE2000)를 비교 (같은 시험지로 비교해야 공정)
#   - 보정용 사진만 쓴다 (평가용은 열지 않음). color_matrix.json은 바꾸지 않는다
#   결과: 화면 출력만 (decisions.md D9 · data_notes에 숫자를 옮겨 적음)
import json
import numpy as np
import colour
from common import ROOT, load_gehler_index
from pipeline import XYZ_STD, D50, apply_illuminant, normalize_exposure, fit_matrix

SUSPECT = "8D5U5541.png"

d = load_gehler_index("calib")
wb = normalize_exposure(apply_illuminant(d["patches"], d["illum"]))   # 08과 똑같은 처리 (정답 조명 · 노출)
sel = d["cams"] == "1D"
names = d["names"][sel]
A = wb[sel]                                             # (29, 24, 3)
assert SUSPECT in names, f"{SUSPECT}가 1D 보정용에 없음"
keep = names != SUSPECT
print(f"1D 보정용 {len(names)}장 (의심 사진 1장 포함) → 뺀 쪽은 {keep.sum()}장")


def fit(a):
    """(사진 수, 24, 3) → 3×3 행렬 (08과 같은 최소제곱)"""
    M, _ = fit_matrix(a.reshape(-1, 3), np.tile(XYZ_STD, (len(a), 1)))
    return M


def delta_e(a, M):
    """같은 칸들에서 행렬로 바꾼 색 vs 공식 표준 색의 ΔE2000 (빈칸 제외)"""
    rgb = a.reshape(-1, 3)
    std = np.tile(XYZ_STD, (len(a), 1))
    ok = ~np.isnan(rgb).any(axis=1)
    lab_fit = colour.XYZ_to_Lab(rgb[ok] @ M.T, D50)
    lab_std = colour.XYZ_to_Lab(std[ok], D50)
    return colour.delta_E(lab_fit, lab_std, method="CIE 2000")


M_all = fit(A)                                          # 지금 쓰는 행렬 (29장)
M_wo = fit(A[keep])                                     # 의심 사진을 뺀 행렬 (28장)

# 확인: 29장으로 맞춘 행렬이 color_matrix.json의 1D 행렬과 같아야 함 (같은 처리라는 증거)
saved = np.array(json.loads((ROOT / "reports" / "color_matrix.json").read_text())["1D"]["M"])
diff = np.abs(M_all - saved).max()
print(f"color_matrix.json 1D 행렬과 차이(최대): {diff:.6f}  ← 0에 가까우면 08과 같은 처리")

# ① 같은 28장에서 두 행렬 비교 (의심 사진을 뺀 공정한 비교)
e_all, e_wo = delta_e(A[keep], M_all), delta_e(A[keep], M_wo)
print(f"\n[같은 28장 · {len(e_all)}칸에서 ΔE2000]")
print(f"  지금 행렬(29장)      : 중앙값 {np.median(e_all):.3f} · 평균 {e_all.mean():.3f}")
print(f"  의심 사진 뺀 행렬(28장): 중앙값 {np.median(e_wo):.3f} · 평균 {e_wo.mean():.3f}")
print(f"  → 차이: 중앙값 {np.median(e_wo) - np.median(e_all):+.3f} · 평균 {e_wo.mean() - e_all.mean():+.3f}")

# ② 행렬 숫자 자체가 얼마나 바뀌었나
print(f"\n[행렬 9개 값의 변화] 최대 {np.abs(M_wo - M_all).max():.4f} "
      f"(값의 크기는 대략 0.0~1.0)")

# ③ 참고: 의심 사진 자체의 ΔE (두 행렬 모두)
e_s_all, e_s_wo = delta_e(A[~keep], M_all), delta_e(A[~keep], M_wo)
print(f"\n[참고] {SUSPECT} 자체의 ΔE 중앙값: 지금 행렬 {np.median(e_s_all):.2f} · 뺀 행렬 {np.median(e_s_wo):.2f}")
print("\n※ D9: 이 결과와 상관없이 처리 흐름은 바꾸지 않는다 (숫자는 근거로만 기록)")