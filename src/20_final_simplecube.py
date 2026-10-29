# src/20_final_simplecube.py
# 목적: SimpleCube++ test 462장으로 조명 추정 정확도(각도 오차)를 최종 1회 평가한다 (보조 결과, D1 · D5 · D10)
#   - 방법: Gray World(p=1) · White Patch(p=∞) · Shades of Gray(p = train에서 정한 값, D11) — 큐브 자리 · 포화 제외
#   - 설정은 모두 train에서 정해 reports/illum_params.json에 고정되어 있다. test로는 아무것도 고르지 않는다
#   - 오차 범위: 사진 단위 부트스트랩 2,000회 · 95% · seed 42 (카메라 1종이라 층화 없음, D10)
#     방법 비교 = 같은 재표본에서 "중앙값 차이" (음수면 앞쪽 방법이 더 정확)
#   실행 (둘 중 하나를 꼭 적어야 돈다)
#     python src/20_final_simplecube.py --rehearsal   ← train 1,772장으로 리허설. 14 · 15와 같은 숫자가 나와야 정상 (test 안 엶)
#     python src/20_final_simplecube.py --final       ← test 최종 1회. 커밋 안 된 변경이 있거나 이미 실행했으면 멈춤
#   결과: --final     → reports/final/simplecube_test.md (깃에 올리는 최종 기록) + outputs/final/simplecube_test.csv
#         --rehearsal → outputs/final_rehearsal/simplecube_train.md · .csv
import sys
import time
import numpy as np
import pandas as pd
from common import ROOT, SC_DIR, angular_error, load_simplecube, load_simplecube_gt
from illuminant import load_methods, sog_many, summarize
from scoring import boot_median, paired_median_diff, final_lock, stamp

if "--final" in sys.argv:
    mode, split, list_name = "최종 1회 (test)", "test", "simplecube_test.txt"
    out_md = ROOT / "reports" / "final" / "simplecube_test.md"
    out_csv = ROOT / "outputs" / "final" / "simplecube_test.csv"
    commit = final_lock(out_md)                                  # 이미 실행 / 커밋 안 된 변경 → 여기서 멈춤
    print("⚠️ SimpleCube++ test 462장을 엽니다 — 최종 1회 (D5)")
elif "--rehearsal" in sys.argv:
    mode, split, list_name = "리허설 (train, 최종 숫자 아님)", "train", "simplecube_calib.txt"
    out_md = ROOT / "outputs" / "final_rehearsal" / "simplecube_train.md"
    out_csv = out_md.with_suffix(".csv")
    commit = "리허설"
else:
    raise SystemExit("사용법: python src/20_final_simplecube.py --rehearsal  (train 리허설)  |  --final  (test 최종 1회)")
out_md.parent.mkdir(parents=True, exist_ok=True)
out_csv.parent.mkdir(parents=True, exist_ok=True)

methods = load_methods("simplecube")                    # {"gray_world": 1, "white_patch": inf, "shades_of_gray": p}
NAMES = {"gray_world": "Gray World", "white_patch": "White Patch", "shades_of_gray": f"Shades of Gray (p={methods['shades_of_gray']})"}
ps = sorted(set(methods.values()))
ids = (ROOT / "splits" / list_name).read_text().split()
gts = load_simplecube_gt(split)
print(f"SimpleCube++ {split} {len(ids)}장, 방법: " + ", ".join(NAMES.values()))

rows, t0 = [], time.time()
for k, img_id in enumerate(ids, 1):
    lin, valid = load_simplecube(img_id, split)
    est = sog_many(lin, valid, ps)
    row = {"image": img_id, "valid_ratio": round(float(valid.mean()), 3)}
    for m, p in methods.items():
        row[m] = float(angular_error(est[p], gts[img_id]))
    rows.append(row)
    if k % 200 == 0 or k == len(ids):
        print(f"  {k:4d}/{len(ids)} ({time.time() - t0:.0f}초)")

res = pd.DataFrame(rows)
prop = pd.read_csv(SC_DIR / "auxiliary" / f"{split}_properties.csv", dtype={"image": str})
prop["image"] = prop["image"].str.replace(".png", "", regex=False)
res = res.merge(prop[["image", "illumination", "place"]], on="image", how="left")
res.to_csv(out_csv, index=False)

# ── 보고 ──────────────────────────────────────────────────────
L = [f"# SimpleCube++ 조명 추정 정확도 — {mode}", "",
     stamp(mode, commit),
     f"- 사진 {len(res)}장 · 큐브 자리 · 포화 제외 · 설정은 train에서 고정 (`reports/illum_params.json`)",
     "- 오차 범위: 사진 단위 부트스트랩 2,000회 · 95% 백분위 · seed 42", "",
     "## 각도 오차 (도)", "",
     "| 방법 | 중앙값 [95% 구간] | 평균 | trimean | 좋은 25% | 나쁜 25% | 최대 |", "|---|---|---|---|---|---|---|"]
for m in methods:
    s = summarize(res[m])
    lo, hi = boot_median(res[m])
    L.append(f"| {NAMES[m]} | **{s['median']:.2f}** [{lo:.2f}, {hi:.2f}] | {s['mean']:.2f} | {s['trimean']:.2f} | "
             f"{s['best25']:.2f} | {s['worst25']:.2f} | {s['max']:.2f} |")

L += ["", "## 방법 비교 — 중앙값 차이 (같은 재표본, 음수면 앞쪽이 더 정확)", "",
      "| 비교 | 차이 [95% 구간] | 판단 |", "|---|---|---|"]
for a, b in [("shades_of_gray", "gray_world"), ("shades_of_gray", "white_patch"), ("gray_world", "white_patch")]:
    d, lo, hi = paired_median_diff(res[a], res[b])
    verdict = "0 포함 안 함 → 의미 있는 차이" if lo > 0 or hi < 0 else "0 포함 → 차이 있다고 말할 수 없음"
    L.append(f"| {NAMES[a]} − {NAMES[b]} | {d:+.2f}° [{lo:+.2f}, {hi:+.2f}] | {verdict} |")

L += ["", "## 조명 그룹별 중앙값 (도) [95% 구간]", "",
      "라벨이 unknown · mixed인 사진은 그룹에서 빠진다 (D1)", "",
      "| 그룹 | 사진 수 | " + " | ".join(NAMES.values()) + " |", "|---|---|" + "---|" * len(methods)]
for col, groups in [("illumination", ["natural", "artificial"]), ("place", ["outdoor", "indoor"])]:
    for g in groups:
        sub = res[res[col] == g]
        if len(sub) == 0:
            L.append(f"| {col} = {g} | 0 | " + " | ".join("-" for _ in methods) + " |")
            continue
        cells = []
        for m in methods:
            lo, hi = boot_median(sub[m])
            cells.append(f"{sub[m].median():.2f} [{lo:.2f}, {hi:.2f}]")
        L.append(f"| {col} = {g} | {len(sub)} | " + " | ".join(cells) + " |")
L += ["", f"쓸 수 있는 픽셀 비율: 중앙값 {res['valid_ratio'].median():.0%} · 최저 {res['valid_ratio'].min():.0%}"]

out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n" + "\n".join(L))
if split == "train":
    print("\n※ 리허설 확인: 중앙값이 14 · 15 출력과 같아야 정상 — Gray World 1.91 · Shades of Gray 1.45 · White Patch 2.44")
print(f"\n저장: {out_md.relative_to(ROOT)}, {out_csv.relative_to(ROOT)}")