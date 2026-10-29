# src/15_sog_simplecube.py
# 목적: Shades of Gray의 설정값 p를 SimpleCube++ 보정용(train 1,772장)으로 정한다 (D11)
#   - 후보 p = 1~12 (p=1은 Gray World와 같음). 사진마다 같은 픽셀로 13가지(1~12 + ∞=White Patch)를 한 번에 계산
#   - 기준: 각도 오차 "중앙값"이 가장 작은 p (같으면 작은 p). 색 이름 정답률로 고르지 않는다 (D11)
#   - 여기서 고른 p는 SimpleCube++ 각도 오차 평가에만 쓴다. 색 이름 채점에 쓸 p는 Gehler 보정용으로 따로 (16)
#   - test 462장은 열지 않는다 (D5)
#   결과: reports/illum_params.json의 "simplecube" (고른 p + 후보별 표, 깃에 올림)
#         outputs/illum/sog_simplecube_train.csv (사진별 · p별 오차)
import time
import numpy as np
import pandas as pd
from common import ROOT, SC_DIR, angular_error, load_simplecube, load_simplecube_gt
from illuminant import P_CANDIDATES, sog_many, choose_p, save_p, summarize

OUT = ROOT / "outputs" / "illum"
OUT.mkdir(parents=True, exist_ok=True)
PS = P_CANDIDATES + [np.inf]

ids = (ROOT / "splits" / "simplecube_calib.txt").read_text().split()
gts = load_simplecube_gt("train")
print(f"SimpleCube++ train {len(ids)}장, 후보 p = {P_CANDIDATES[0]}~{P_CANDIDATES[-1]} (+ ∞ = White Patch, 참고)")

err = {p: [] for p in PS}
t0 = time.time()
for k, img_id in enumerate(ids, 1):
    lin, valid = load_simplecube(img_id, "train")
    est = sog_many(lin, valid, PS)
    for p in PS:
        err[p].append(float(angular_error(est[p], gts[img_id])))
    if k % 300 == 0 or k == len(ids):
        print(f"  {k:4d}/{len(ids)} ({time.time() - t0:.0f}초)")

# 사진별 결과 저장
res = pd.DataFrame({"image": ids, **{("p_inf" if np.isinf(p) else f"p{p:02d}"): err[p] for p in PS}})
res.to_csv(OUT / "sog_simplecube_train.csv", index=False)

# ── 후보별 표 ───────────────────────────────────────────────────
best, meds = choose_p(err)
print("\n  p   중앙값   평균  trimean  나쁜25%")
for p in PS:
    s = summarize(err[p])
    tag = "  ← Gray World" if p == 1 else "  ← White Patch" if np.isinf(p) else ""
    star = " ★" if p == best else "  "
    print(f"{'∞' if np.isinf(p) else p:>3}{star} {s['median']:6.2f} {s['mean']:6.2f} {s['trimean']:7.2f} {s['worst25']:8.2f}{tag}")

s_best, s_gw = summarize(err[best]), summarize(err[1])
print(f"\n고른 p = {best}  (중앙값 {s_best['median']:.2f}° · Gray World {s_gw['median']:.2f}° · "
      f"White Patch {summarize(err[np.inf])['median']:.2f}°)")

# 조명 그룹별로 고른 p가 어떤지 (참고)
prop = pd.read_csv(SC_DIR / "auxiliary" / "train_properties.csv", dtype={"image": str})
prop["image"] = prop["image"].str.replace(".png", "", regex=False)
g = res.merge(prop[["image", "illumination", "place"]], on="image", how="left")
col_best = f"p{best:02d}"
print("\n[조명 그룹별 중앙값] Gray World / 고른 p / White Patch")
for col, groups in [("illumination", ["natural", "artificial"]), ("place", ["outdoor", "indoor"])]:
    for grp in groups:
        sub = g[g[col] == grp]
        print(f"  {col:12s} {grp:10s} ({len(sub):4d}장)  {sub['p01'].median():5.2f} / "
              f"{sub[col_best].median():5.2f} / {sub['p_inf'].median():5.2f}")

save_p("simplecube", {
    "p": best, "rule": "SimpleCube++ train 각도 오차 중앙값 최소 (후보 1~12, 같으면 작은 p)",
    "n_images": len(ids),
    "median_by_p": {str(p): round(m, 4) for p, m in meds.items()},
    "white_patch_median": round(float(np.nanmedian(err[np.inf])), 4),
})
print(f"\n저장: reports/illum_params.json (simplecube), {(OUT / 'sog_simplecube_train.csv').relative_to(ROOT)}")