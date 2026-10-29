# src/17_check_cct.py
# 목적: (점검) D10 결정 2의 "정답 조명 → 색온도" 계산이 맞게 되는지 확인하고, 조명 그룹을 나눌 행렬을 정한다
#   배경: 우리 행렬(color_matrix.json)은 "조명 이득으로 나눈 뒤"(화이트밸런스 뒤)의 RGB를 XYZ로 바꾸도록 맞춘 것이다.
#         정답 조명 자체는 나누기 전 카메라 원래 RGB(보통 초록이 가장 큼)라서, 이 행렬에 넣으면
#         조명 색이 초록 쪽으로 밀린 엉뚱한 좌표가 나올 수 있다. McCamy 식은 흑체 궤적 근처에서만 의미가 있다
#   확인 방법: 흑체 궤적에서 떨어진 정도 |Duv|. 실제 조명은 대부분 0.02 안쪽
#   비교:  A. 우리 행렬 (D10 원안)
#          B. Adobe DNG 카메라 행렬 (LibRaw 공개 표, 사진 앱이 "찍을 때 색온도"를 보여 줄 때 쓰는 종류)
#   1D 사진(8D5U…)의 기종: 데이터 페이지는 "Canon 1D"라고만 쓰지만, PNG가 원본의 절반 크기(2041×1359)라고 밝힌다
#             → 원본 4082×2718 ≈ 1,110만 화소 = EOS-1Ds(유효 4064×2704). 원조 EOS-1D(2464×1648, 400만 화소)로는 안 나오는 크기
#             (5D도 같은 방식으로 맞음: 2193×1460 × 2 = 4386×2920 = LibRaw의 EOS 5D 유효 영역)
#             → EOS-1Ds 행렬을 쓴다. EOS-1D 행렬은 참고로 함께 출력 (|Duv|만으로는 두 기종이 잘 구분되지 않음)
#   멈추는 규칙 (결과 보기 전에 고정): 고른 행렬의 카메라별 |Duv| 중앙값이 0.02를 넘으면 그룹 나누기를 믿기 어려움 → 저장 안 함, 다시 의논
#   - 보정용 190장의 정답 조명만 쓴다. 색 이름 채점 결과는 보지 않는다 (D10: 그룹을 나눠도 결과를 미리 보는 것이 아님)
#   결과: 화면 출력 + reports/illum_params.json "cct" (19 · 최종 평가가 그룹을 나눌 때 읽음)
import numpy as np
from common import load_gehler_index
from illuminant import save_p
from scoring import illum_xy, cct_duv, to_group, GROUPS

DUV_OK = 0.02

d = load_gehler_index("calib")
cams, illum = d["cams"], d["illum"]
print(f"Gehler-Shi 보정용 {len(cams)}장 (1D {np.sum(cams == '1D')} · 5D {np.sum(cams == '5D')}) 정답 조명")
g = illum / illum[:, 1:2]
for cam in ["1D", "5D"]:
    s = cams == cam
    print(f"  {cam} 정답 조명 r/g 중앙값 {np.median(g[s, 0]):.2f} · b/g 중앙값 {np.median(g[s, 2]):.2f}"
          f"  (1보다 작으면 원래 RGB에서 초록이 가장 큼)")

cands = {
    "A. 우리 행렬 (D10 원안)": "ours",
    "B. Adobe 5D + EOS-1D": {"1D": "EOS-1D", "5D": "EOS 5D"},
    "B. Adobe 5D + EOS-1Ds": {"1D": "EOS-1Ds", "5D": "EOS 5D"},
}
res = {}
print(f"\n{'방법':24s} {'카메라':4s} {'x':>6s} {'y':>6s} {'|Duv| 중앙':>10s} {'|Duv|<0.02':>10s}  색온도 중앙(10%~90%)")
for label, how in cands.items():
    xy = illum_xy(illum, cams, how)
    cct, duv = cct_duv(xy)
    res[label] = (how, cct, duv)
    for cam in ["1D", "5D"]:
        s = cams == cam
        lo, mid, hi = np.percentile(cct[s], [10, 50, 90])
        print(f"{label:24s} {cam:4s} {np.median(xy[s, 0]):6.3f} {np.median(xy[s, 1]):6.3f} "
              f"{np.median(np.abs(duv[s])):10.4f} {np.mean(np.abs(duv[s]) < DUV_OK):10.0%}  "
              f"{mid:5.0f}K ({lo:.0f}~{hi:.0f})")
print("  참고: 흑체 궤적 위 5000K ≈ (x 0.345, y 0.352) · 6500K ≈ (0.313, 0.324) · 3000K ≈ (0.437, 0.404)")

# ── 고른 행렬 (1D = EOS-1Ds, 사진 크기 근거) · 멈추는 규칙 확인 ─────────────
pick = {"1D": "EOS-1Ds", "5D": "EOS 5D"}
_, cct, duv = res["B. Adobe 5D + EOS-1Ds"]
med = {c: float(np.median(np.abs(duv[cams == c]))) for c in ["1D", "5D"]}
print(f"\n고른 행렬: 1D → EOS-1Ds · 5D → EOS 5D  (|Duv| 중앙값 1D {med['1D']:.4f} · 5D {med['5D']:.4f})")
if max(med.values()) > DUV_OK:
    print(f"⚠️ |Duv| 중앙값이 {DUV_OK}를 넘음 → 그룹 나누기를 믿기 어려움. 저장하지 않고 멈춤 (다시 의논)")
    raise SystemExit(1)
cct_1d = res["B. Adobe 5D + EOS-1D"][1]
print(f"  참고: 1D를 EOS-1D 행렬로 계산하면 그룹이 달라지는 1D 사진 "
      f"{np.sum((to_group(cct_1d) != to_group(cct)) & (cams == '1D'))}장 / {np.sum(cams == '1D')}장")

# 그룹별 사진 수 (A와 B 비교 — A로 나눴다면 얼마나 달랐을지)
grp_a = to_group(res["A. 우리 행렬 (D10 원안)"][1])
grp_b = to_group(cct)
print("\n[조명 그룹별 사진 수]   A(원안) → B(고른 행렬)")
for key, label, _, _ in GROUPS:
    by_cam = " · ".join(f"{c} {np.sum((grp_b == key) & (cams == c))}" for c in ["1D", "5D"])
    print(f"  {label:18s} {np.sum(grp_a == key):4d} → {np.sum(grp_b == key):4d}  ({by_cam})")
print(f"  A와 B의 그룹이 다른 사진: {np.sum(grp_a != grp_b)}장")

save_p("cct", {
    "how": pick,
    "rule": "Adobe DNG 행렬(LibRaw colordata.cpp) 역행렬로 정답 조명 → XYZ → xy → McCamy. "
            "1D 사진은 PNG 크기(원본 4082×2718)로 EOS-1Ds (17_check_cct.py)",
    "median_abs_duv": {c: round(v, 4) for c, v in med.items()},
    "calib_group_counts": {key: int(np.sum(grp_b == key)) for key, _, _, _ in GROUPS},
})
print("\n저장: reports/illum_params.json (cct)")