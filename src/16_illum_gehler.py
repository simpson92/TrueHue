# src/16_illum_gehler.py
# 목적: Gehler-Shi 보정용 190장에서 조명 추정 각도 오차를 재고, 색 이름 채점에 쓸 Shades of Gray p를 정한다 (D11)
#   - 사진마다 포화 픽셀 + 컬러체커 가림 영역(reports/chart_boxes.csv)을 뺀 나머지로만 추정 → 공정성 마스킹 (D8)
#   - 후보 p = 1~12 (+ ∞ = White Patch 참고), 기준은 각도 오차 중앙값 최소 (같으면 작은 p)
#   - 정답 조명: 데이터셋 정답(SFU판). 의심 사진 8D5U5541도 그대로 포함 (D9 처리 A: 보정 단계는 그대로)
#   - 평가용 378장은 열지 않는다 (D5)
#   결과: reports/illum_params.json의 "gehler" (고른 p + 후보별 표, 깃에 올림)
#         outputs/illum/gehler_calib.csv (사진별 · p별 오차, 쓸 수 있는 픽셀 비율)
import time
import numpy as np
import pandas as pd
from common import ROOT, angular_error, load_gehler_index, load_gehler_image
from illuminant import P_CANDIDATES, sog_many, choose_p, save_p, summarize

OUT = ROOT / "outputs" / "illum"
OUT.mkdir(parents=True, exist_ok=True)
PS = P_CANDIDATES + [np.inf]

d = load_gehler_index("calib")
n = len(d["names"])
print(f"Gehler-Shi 보정용 {n}장 (1D {np.sum(d['cams'] == '1D')} · 5D {np.sum(d['cams'] == '5D')}), "
      f"후보 p = {P_CANDIDATES[0]}~{P_CANDIDATES[-1]} (+ ∞ = White Patch, 참고)")

rows, t0 = [], time.time()
for k in range(n):
    name = d["names"][k]
    lin, valid = load_gehler_image(name)                # 포화 · 가림 영역 제외, 블랙 레벨 빼기
    est = sog_many(lin, valid, PS)
    row = {"image": name, "camera": d["cams"][k], "valid_ratio": round(float(valid.mean()), 3)}
    for p in PS:
        row["p_inf" if np.isinf(p) else f"p{p:02d}"] = float(angular_error(est[p], d["illum"][k]))
    rows.append(row)
    if (k + 1) % 20 == 0 or k + 1 == n:
        print(f"  {k + 1:3d}/{n} ({time.time() - t0:.0f}초)")

res = pd.DataFrame(rows)
res.to_csv(OUT / "gehler_calib.csv", index=False)
col = lambda p: "p_inf" if np.isinf(p) else f"p{p:02d}"

# ── 후보별 표 (전체 · 카메라별 중앙값) ───────────────────────────────
best, meds = choose_p({p: res[col(p)].to_numpy() for p in P_CANDIDATES})
print("\n  p   중앙값   평균  trimean  나쁜25%   | 1D 중앙값  5D 중앙값")
for p in PS:
    s = summarize(res[col(p)])
    m1 = res.loc[res.camera == "1D", col(p)].median()
    m5 = res.loc[res.camera == "5D", col(p)].median()
    tag = "  ← Gray World" if p == 1 else "  ← White Patch" if np.isinf(p) else ""
    star = " ★" if p == best else "  "
    print(f"{'∞' if np.isinf(p) else p:>3}{star} {s['median']:6.2f} {s['mean']:6.2f} {s['trimean']:7.2f} "
          f"{s['worst25']:8.2f}   | {m1:8.2f}  {m5:9.2f}{tag}")

print(f"\n고른 p = {best}  (중앙값 {res[col(best)].median():.2f}° · Gray World {res['p01'].median():.2f}° · "
      f"White Patch {res['p_inf'].median():.2f}°)")
print(f"쓸 수 있는 픽셀 비율: 중앙값 {res['valid_ratio'].median():.0%} · 가장 적은 사진 {res['valid_ratio'].min():.0%} "
      f"({res.loc[res['valid_ratio'].idxmin(), 'image']})")

# 오차가 가장 큰 사진 (나중에 직접 볼 후보)
worst = res.nlargest(5, col(best))[["image", "camera", "valid_ratio", "p01", col(best), "p_inf"]]
print(f"\n[고른 p 기준 오차가 큰 5장]\n{worst.round(2).to_string(index=False)}")
print(f"\n[참고] 의심 사진 8D5U5541: " + ", ".join(
    f"{c} {res.loc[res.image == '8D5U5541.png', c].iloc[0]:.2f}°" for c in ["p01", col(best), "p_inf"]))

save_p("gehler", {
    "p": best, "rule": "Gehler-Shi 보정용 190장 각도 오차 중앙값 최소 (후보 1~12, 같으면 작은 p, 가림 영역 적용)",
    "n_images": n,
    "median_by_p": {str(p): round(m, 4) for p, m in meds.items()},
    "white_patch_median": round(float(res["p_inf"].median()), 4),
})
print(f"\n저장: reports/illum_params.json (gehler), {(OUT / 'gehler_calib.csv').relative_to(ROOT)}")