# src/14_illum_simplecube.py
# 목적: 조명 추정 2종(Gray World · White Patch)을 SimpleCube++ 보정용(train 1,772장)에서 돌려 각도 오차를 잰다
#   - 각도 오차: 추정한 조명 색과 정답 조명 색의 "방향" 차이(도). 밝기는 보지 않는다 (common.angular_error)
#   - 사진마다: 큐브 자리 · 포화 픽셀을 뺀 나머지(common.load_simplecube의 valid)로만 추정 → 공정성 마스킹
#   - test 462장은 열지 않는다 (최종 평가 때 한 번만, D5)
#   - Shades of Gray는 p를 정하는 단계에서 따로 한다 (D11). 여기서는 설정값이 없는 두 방법만
#   결과: outputs/illum/simplecube_train.csv (사진별 추정 조명 · 오차), 화면에 요약 · 조명 그룹별 요약
import time
import numpy as np
import pandas as pd
from common import ROOT, SC_DIR, angular_error, load_simplecube, load_simplecube_gt
from illuminant import METHODS, shades_of_gray, summarize

OUT = ROOT / "outputs" / "illum"
OUT.mkdir(parents=True, exist_ok=True)

ids = (ROOT / "splits" / "simplecube_calib.txt").read_text().split()    # train = 보정용
gts = load_simplecube_gt("train")
print(f"SimpleCube++ train {len(ids)}장 (test는 열지 않음)")

rows, t0 = [], time.time()
for k, img_id in enumerate(ids, 1):
    lin, valid = load_simplecube(img_id, "train")       # 큐브 자리 → 포화 → 블랙 레벨 순서 (common)
    row = {"image": img_id, "valid_ratio": round(float(valid.mean()), 3)}
    for name, p in METHODS.items():
        est = shades_of_gray(lin, valid, p)
        row[f"{name}_err"] = float(angular_error(est, gts[img_id]))
        row.update({f"{name}_{c}": est[j] for j, c in enumerate("rgb")})
    rows.append(row)
    if k % 200 == 0 or k == len(ids):
        print(f"  {k:4d}/{len(ids)} ({time.time() - t0:.0f}초)")

res = pd.DataFrame(rows)

# 조명 그룹 라벨 붙이기 (D1: 자연광/인공, 실외/실내. unknown · mixed는 그룹 평가에서 제외)
prop = pd.read_csv(SC_DIR / "auxiliary" / "train_properties.csv", dtype={"image": str})
prop["image"] = prop["image"].str.replace(".png", "", regex=False)
res = res.merge(prop[["image", "illumination", "place"]], on="image", how="left")
res.to_csv(OUT / "simplecube_train.csv", index=False)

# ── 요약 ───────────────────────────────────────────────────────
cols = ["n", "mean", "median", "trimean", "best25", "worst25", "max"]
print("\n[전체] 각도 오차(도) — 중앙값이 주로 보는 숫자")
table = pd.DataFrame({name: summarize(res[f"{name}_err"]) for name in METHODS}).T[cols]
print(table.round(2).to_string())

print("\n[조명 그룹별] 중앙값(도), 괄호는 사진 수")
for col, groups in [("illumination", ["natural", "artificial"]), ("place", ["outdoor", "indoor"])]:
    for g in groups:
        sub = res[res[col] == g]
        cells = "  ".join(f"{name} {np.median(sub[f'{name}_err']):5.2f}" for name in METHODS)
        print(f"  {col:12s} {g:10s} ({len(sub):4d}장)  {cells}")

# 두 방법 중 사진마다 어느 쪽이 더 잘 맞았나 (서로 약점이 다른지)
gw_better = (res["gray_world_err"] < res["white_patch_err"]).mean()
print(f"\nGray World가 White Patch보다 잘 맞은 사진 비율: {gw_better:.0%}")
print(f"쓸 수 있는 픽셀 비율: 중앙값 {res['valid_ratio'].median():.0%} (큐브 자리 약 15.6% + 포화 픽셀이 빠짐)")
print(f"\n저장: {(OUT / 'simplecube_train.csv').relative_to(ROOT)}")