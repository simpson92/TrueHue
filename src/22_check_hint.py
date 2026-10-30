# src/22_check_hint.py
# 목적: 웹 화면 "○○에 가까움" 표시의 기준 여유를 보정용 190장으로 정한다 (D13, 웹-3)
#   - 평가와 같은 처리(Shades of Gray p · 컬러체커로 밝기 · 카메라 행렬)로 보정용 24칸에 이름을 붙이고,
#     칸마다 규칙 경계까지의 여유(채도 · 색상 각도 · 밝기)와 맞았는지를 같이 본다
#   - 고르는 규칙 (표를 보기 전에 고정):
#       경계 종류마다, 여유가 t 미만인 칸들의 틀린 비율이 30% 이상인 가장 큰 t (그런 칸이 20개 이상일 때)
#       = "표시가 붙은 칸은 열에 셋 이상 실제로 틀린다" — 전체 틀린 비율(보정용 약 14%)의 두 배 남짓
#       그런 t가 없으면 그 종류는 표시하지 않음 (t = 0)
#   - 이름 규칙 · 평가 결과는 바꾸지 않는다. 이 숫자는 웹 화면 표시에만 쓴다. 평가용 사진은 쓰지 않는다
#   필요: 18 (outputs/illum/gehler_est_calib.csv)
#   결과: 화면 출력 + reports/web_params.json "hint" (웹 화면이 읽음, 깃에 올림)
import json
import numpy as np
import pandas as pd
from common import ROOT, load_gehler_index
from illuminant import load_methods
from pipeline import load_color_matrices, patches_to_lab, name_patches
from scoring import judge, ALLOWED
from hint import neighbors, near_name, KINDS

ERR_MIN, N_MIN = 0.30, 20
GRID = {"chroma": np.arange(0.5, 10.01, 0.5), "hue": np.arange(1, 20.01, 1.0), "L": np.arange(1, 15.01, 1.0)}
UNIT = {"chroma": "", "hue": "°", "L": ""}
OUT = ROOT / "reports" / "web_params.json"

d = load_gehler_index("calib")
p = load_methods("gehler")["shades_of_gray"]
est = pd.read_csv(ROOT / "outputs" / "illum" / "gehler_est_calib.csv").set_index("image").loc[d["names"]]
illum = est[[f"shades_of_gray_{c}" for c in "rgb"]].to_numpy()
lab = patches_to_lab(d["patches"], illum, d["cams"], load_color_matrices())
names = name_patches(lab)
correct, _ = judge(names, d["patches"])

rows = []
for i, k in zip(*np.nonzero(~np.isnan(correct))):
    row = {"patch": k + 1, "name": names[i, k], "correct": bool(correct[i, k]), "lab": lab[i, k]}
    for kind, m, other in neighbors(lab[i, k]):
        row[f"{kind}_m"], row[f"{kind}_n"] = m, other
    rows.append(row)
df = pd.DataFrame(rows)
err_all = 1 - df["correct"].mean()
print(f"보정용 {len(d['names'])}장 · {len(df)}칸, Shades of Gray (p = {p}) — 정답률 {100 * (1 - err_all):.1f}% "
      f"(19의 SoG와 같아야 정상: 86.0%)")

# ── 경계 종류마다 여유 t별 표 → 규칙대로 t 고르기 ──────────────────
thr = {}
for kind in KINDS:
    sub = df.dropna(subset=[f"{kind}_m"])
    print(f"\n[{kind} 경계] 이 경계가 있는 칸 {len(sub)}개 · 그중 틀린 비율 {100 * (1 - sub['correct'].mean()):.1f}%")
    print(f"   t{UNIT[kind]:1s}   여유<t 칸 수   그중 틀림   (여유≥t 틀림)")
    best = 0.0
    for t in GRID[kind]:
        flag = sub[f"{kind}_m"] < t
        n = int(flag.sum())
        e_in = 1 - sub.loc[flag, "correct"].mean() if n else np.nan
        e_out = 1 - sub.loc[~flag, "correct"].mean() if (~flag).sum() else np.nan
        ok = n >= N_MIN and e_in >= ERR_MIN
        if ok:
            best = float(t)
        print(f"  {t:5.1f}   {n:9d}   {100 * e_in:7.1f}%   ({100 * e_out:5.1f}%)" + ("  ✓" if ok else ""))
    thr[kind] = best
    print(f"  → 고른 기준: {kind} 여유 < {best}{UNIT[kind]}" + ("" if best else " (표시 안 함)"))

# ── 고른 기준으로 합쳐 보기 ────────────────────────────────────────
near = [near_name(r["lab"], thr) for _, r in df.iterrows()]
df["near"] = near
flag = df["near"].notna()
wrong = ~df["correct"]
print(f"\n[고른 기준 {thr}]")
print(f"  표시가 붙는 칸: {flag.mean():.0%} ({flag.sum()} / {len(df)})")
print(f"  틀린 비율 — 표시 붙은 칸 {100 * wrong[flag].mean() if flag.any() else float('nan'):.1f}% · 안 붙은 칸 {100 * wrong[~flag].mean():.1f}% (전체 {100 * err_all:.1f}%)")
print(f"  전체 오답 중 표시가 붙은 비율: {wrong[flag].sum() / wrong.sum():.0%} ({wrong[flag].sum()} / {wrong.sum()})")
helpful = [r["near"] in ALLOWED[r["patch"] - 1] for _, r in df[flag & wrong].iterrows()]
if helpful:
    print(f"  표시가 붙고 틀린 칸에서, 함께 보여 준 이름이 정답인 비율: {np.mean(helpful):.0%} ({sum(helpful)} / {len(helpful)})")

# 칸별로 표시가 얼마나 붙나 (회색 · 흰색 칸 확인)
by = df.groupby("patch").agg(n=("correct", "size"), acc=("correct", "mean"), flagged=("near", lambda s: s.notna().mean()))
print("\n[칸별] 정답률 · 표시 붙는 비율 (흰색 · 회색 칸과 2번이 많이 붙어야 자연스러움)")
print("  " + "  ".join(f"{k}:{100 * r.acc:.0f}/{100 * r.flagged:.0f}" for k, r in by.iterrows()))

params = json.loads(OUT.read_text()) if OUT.exists() else {}
params["hint"] = {
    **{k: round(v, 2) for k, v in thr.items()},
    "rule": f"보정용 190장 SoG(p={p}) 결과에서, 여유 < t인 칸의 틀린 비율이 {int(ERR_MIN * 100)}% 이상인 가장 큰 t "
            f"(칸 {N_MIN}개 이상). 웹 표시용, 이름 규칙 · 평가와 무관 (22_check_hint.py)",
    "flagged_share": round(float(flag.mean()), 3),
    "error_flagged": round(float(wrong[flag].mean()), 3) if flag.any() else None,
    "error_unflagged": round(float(wrong[~flag].mean()), 3),
}
OUT.write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\n")
print(f"\n저장: {OUT.relative_to(ROOT)} (hint)")