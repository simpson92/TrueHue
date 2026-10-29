# src/19_score_calib.py
# 목적: 보정용 190장으로 색 이름 채점을 "연습"한다 — 기준선 · 보정 3종 · 상한선 (D6 · D10)
#   - 처리 흐름은 모든 방법이 똑같고 "조명 색"만 다르다 (pipeline.patches_to_lab)
#   - 채점 · 보고 계산은 scoring.py 부품 → 최종 평가 때도 같은 부품을 쓴다
#   - 왜 연습이 필요한가: 평가용은 한 번만 열 수 있으니(D5), 계산이 맞는지 · 숫자가 말이 되는지 여기서 먼저 확인
#   - 주의: 보정용은 행렬(08) · 기준선 · p를 정할 때 쓴 사진이다 → 여기 숫자는 평가용보다 좋게 나올 수 있다 (최종 숫자 아님)
#   - 규칙 · 정답표 · 설정은 여기 결과를 보고 바꾸지 않는다 (D2 · D4 · D11 확정)
#   필요: 17(조명 그룹 행렬) · 18(추정 조명 · 기준선)을 먼저 실행
#   결과: 화면 출력 + outputs/scores/calib_patches.csv (사진 · 칸 · 방법별 이름 · 정답 여부)
import json
import numpy as np
import pandas as pd
from common import ROOT, NEUTRALS, load_gehler_index
from illuminant import PARAMS, load_methods
from pipeline import load_color_matrices, patches_to_lab, name_patches
from scoring import (judge, per_photo, rate, paired_diff, bootstrap_rate, illum_groups,
                     SUBSETS, GROUPS, REF_NAMES)

OUT = ROOT / "outputs" / "scores"
OUT.mkdir(parents=True, exist_ok=True)

d = load_gehler_index("calib")
names_img, cams, P = d["names"], d["cams"], d["patches"]
n = len(names_img)
params = json.loads(PARAMS.read_text())
est = pd.read_csv(ROOT / "outputs" / "illum" / "gehler_est_calib.csv").set_index("image").loc[names_img]
p_sog = load_methods("gehler")["shades_of_gray"]

# ── 방법별 조명 색 ────────────────────────────────────────────
ILLUM = {
    "기준선(고정)": np.array([params["baseline"][c] for c in cams]),
    "Gray World": est[[f"gray_world_{c}" for c in "rgb"]].to_numpy(),
    "White Patch": est[[f"white_patch_{c}" for c in "rgb"]].to_numpy(),
    f"SoG(p={p_sog})": est[[f"shades_of_gray_{c}" for c in "rgb"]].to_numpy(),
    "상한선(정답)": d["illum"],
}
BASE, UPPER = "기준선(고정)", "상한선(정답)"
CORR = [m for m in ILLUM if m not in (BASE, UPPER)]

# ── 기본 확인 ─────────────────────────────────────────────────
no_gray = int(np.isnan(P[:, [k - 1 for k in NEUTRALS], :]).any(axis=2).all(axis=1).sum())
n_q = int((~np.isnan(P).any(axis=2)).sum())
print(f"Gehler-Shi 보정용 {n}장 (1D {np.sum(cams == '1D')} · 5D {np.sum(cams == '5D')}), 채점 문제 {n_q}개")
print(f"회색 칸 20~23번이 모두 빈 사진: {no_gray}장  (0이 아니면 노출 맞추기가 NaN → 확인 필요, B3)")
groups, cct = illum_groups(d["illum"], cams)
print("조명 그룹: " + " · ".join(f"{label} {np.sum(groups == key)}장" for key, label, _, _ in GROUPS))

# ── 채점 ──────────────────────────────────────────────────────
mats = load_color_matrices()
R = {}
for m, il in ILLUM.items():
    lab = patches_to_lab(P, il, cams, mats)
    nm = name_patches(lab)
    correct, consist = judge(nm, P)
    R[m] = {"names": nm, "lab": lab, "correct": correct, "consist": consist}

# 열: 보고 범위(D10) — 칸 묶음 · 카메라 · 조명 그룹
cols = [(label, ks, None) for label, ks in SUBSETS.items()]
cols += [(cam, None, cams == cam) for cam in ["1D", "5D"]]
cols += [(label.split()[0], None, groups == key) for key, label, _, _ in GROUPS]


def table(metric):
    head = f"{'방법':14s}" + "".join(f"{c[0]:>12s}" for c in cols)
    lines = [head]
    for m in ILLUM:
        cells = []
        for _, ks, sel in cols:
            h, t = per_photo(R[m][metric], ks)
            cells.append(f"{rate(h, t, sel):11.1f}%")
        lines.append(f"{m:14s}" + "".join(cells))
    # 문제 수 (사진 수)
    cnt = []
    for _, ks, sel in cols:
        _, t = per_photo(R[BASE]["correct"], ks)
        cnt.append(f"{int(t[sel].sum() if sel is not None else t.sum()):>7d}({int(sel.sum()) if sel is not None else n:>3d})")
    lines.append(f"{'문제 수(사진)':14s}" + "".join(f"{c:>12s}" for c in cnt))
    return "\n".join(lines)


print("\n[정답률] 정답 묶음 {answer, answer_alt} 안이면 정답")
print(table("correct"))
print("\n[이름 일관성] 공식 표준 색에 규칙을 넣은 이름과 같은 비율 (확실 칸은 정답률과 같음)")
print(table("consist"))

# ── 칸별 정답률 · 가장 많이 나온 오답 ─────────────────────────────
print("\n[칸별 정답률 %]  (오답 중 가장 많은 이름: 상한선 · SoG 기준)")
print(f"{'칸':>3s} {'기준 이름':6s}" + "".join(f"{m:>12s}" for m in ILLUM) + "   오답(상한선)  오답(SoG)")
for k in range(24):
    cells = []
    for m in ILLUM:
        c = R[m]["correct"][:, k]
        cells.append(f"{100 * np.nanmean(c):11.0f}%")
    def top_wrong(m):
        c, nm = R[m]["correct"][:, k], R[m]["names"][:, k]
        w = pd.Series([x if x is not None else "(없음)" for x in nm[c == 0]])
        return f"{w.value_counts().index[0]} {w.value_counts().iloc[0]}" if len(w) else "-"
    print(f"{k + 1:3d} {REF_NAMES[k]:6s}" + "".join(cells) + f"   {top_wrong(UPPER):12s}  {top_wrong(CORR[-1])}")

# 회색 칸 채도: D4 예측 확인 (조명 1° 오차당 약 2.4)
print("\n[회색 칸 21 · 22번 채도 중앙값] 기준 10보다 크면 유채색으로 틀림")
for m in ILLUM:
    lab = R[m]["lab"][:, [20, 21], :]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    print(f"  {m:14s} {np.nanmedian(chroma):5.1f}  (10 넘는 비율 {np.nanmean(chroma > 10):4.0%})")

# ── 주 결과 연습: 보정 3종 − 기준선, 전체 24칸 (짝지은 부트스트랩, D10 결정 3 · 4) ──────
print("\n[주 결과 연습 — 보정용] 보정 3종 − 기준선, 전체 24칸 정답률 차이 (95% 구간, 사진 단위 2,000회, 카메라별 뽑기)")
hb, tot = per_photo(R[BASE]["correct"])
for m in CORR + [UPPER]:
    hm, _ = per_photo(R[m]["correct"])
    diff, lo, hi = paired_diff(hm, hb, tot, cams)
    mark = "0 포함 안 함 → 의미 있는 차이" if lo > 0 or hi < 0 else "0 포함 → 차이 있다고 말할 수 없음"
    print(f"  {m:14s} − 기준선: {diff:+5.1f}%p  [{lo:+5.1f}, {hi:+5.1f}]  {mark}")
for m in [BASE] + CORR + [UPPER]:
    h, _ = per_photo(R[m]["correct"])
    lo, hi = bootstrap_rate(h, tot, cams)
    print(f"  {m:14s} 정답률 {rate(h, tot):5.1f}%  [{lo:5.1f}, {hi:5.1f}]")
print("※ 보정용 연습 숫자다. 최종 보고는 평가용 378장(의심 2장 제외 376장도 함께, D9)")

# ── 저장: 사진 · 칸 · 방법별 ──────────────────────────────────────
rows = []
for m in ILLUM:
    nm, c = R[m]["names"], R[m]["correct"]
    for i in range(n):
        for k in range(24):
            if np.isnan(c[i, k]):
                continue
            rows.append({"image": names_img[i], "camera": cams[i], "group": groups[i], "cct": round(float(cct[i])),
                         "patch": k + 1, "method": m, "name": nm[i, k],
                         "correct": int(c[i, k]), "consistent": int(R[m]["consist"][i, k])})
pd.DataFrame(rows).to_csv(OUT / "calib_patches.csv", index=False)
print(f"\n저장: {(OUT / 'calib_patches.csv').relative_to(ROOT)} ({len(rows)}줄)")