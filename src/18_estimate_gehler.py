# src/18_estimate_gehler.py
# 목적: 색 이름 채점에 넣을 "조명 색"을 사진마다 만들어 저장한다 (기준선 + 보정 3종)
#   - 기준선 (D11 결정 1): 카메라별로 보정용 정답 조명을 길이 1로 맞춰 평균 낸 방향 하나 → 모든 사진에 같은 값
#     = "조명에 적응하지 않는 고정 보정". 보정용에서 한 번 정해 reports/illum_params.json "baseline"에 저장
#   - 보정 3종: Gray World(p=1) · White Patch(p=∞) · Shades of Gray(p는 16이 정한 값) — 가림 영역 · 포화 제외 (D8 · D11)
#   - 상한선(정답 조명)은 저장할 필요가 없다 (gehler_index.csv에 이미 있음)
#   - 기본은 보정용 190장. 평가용 378장은 최종 평가 때 딱 한 번:  python src/18_estimate_gehler.py --final  (D5)
#   결과: outputs/illum/gehler_est_calib.csv (최종 평가 때는 gehler_est_test.csv)
#         확인용: 각도 오차 중앙값이 16과 같아야 한다 (같은 사진 · 같은 계산)
import json
import sys
import time
import numpy as np
import pandas as pd
from common import ROOT, angular_error, load_gehler_index, load_gehler_image
from illuminant import load_methods, sog_many, save_p, summarize, PARAMS

OUT = ROOT / "outputs" / "illum"
OUT.mkdir(parents=True, exist_ok=True)
split = "test" if "--final" in sys.argv else "calib"
if split == "test":
    print("⚠️ 평가용 378장을 엽니다 — 최종 평가 1회 (D5). 규칙 · 설정은 모두 고정된 상태여야 함")

# ── 기준선: 보정용 정답 조명으로만 정함 (평가용 실행 때도 보정용 값을 그대로 읽음) ─────
if split == "calib":
    c = load_gehler_index("calib")
    base = {}
    for cam in ["1D", "5D"]:
        u = c["illum"][c["cams"] == cam]
        u = u / np.linalg.norm(u, axis=1, keepdims=True)       # 길이 1로 (밝게 찍힌 사진이 끌고 가지 않게)
        m = u.mean(axis=0)
        base[cam] = m / np.linalg.norm(m)
    save_p("baseline", {
        "rule": "카메라별 보정용 정답 조명을 길이 1로 맞춘 뒤 평균 낸 방향 (D11 결정 1)",
        **{cam: [round(float(v), 6) for v in base[cam]] for cam in base},
        "n_images": {cam: int(np.sum(c["cams"] == cam)) for cam in base},
    })
base = json.loads(PARAMS.read_text())["baseline"]
print("기준선 조명 (R, G, B): " + " · ".join(f"{cam} {np.round(base[cam], 4).tolist()}" for cam in ["1D", "5D"]))

# ── 보정 3종: 사진마다 추정 ─────────────────────────────────────
methods = load_methods("gehler")                     # {"gray_world": 1, "white_patch": inf, "shades_of_gray": p}
ps = sorted(set(methods.values()))
d = load_gehler_index(split)
n = len(d["names"])
print(f"Gehler-Shi {'평가용' if split == 'test' else '보정용'} {n}장, 방법: "
      + ", ".join(f"{k}(p={'∞' if np.isinf(p) else p})" for k, p in methods.items()))

rows, t0 = [], time.time()
for k in range(n):
    name = d["names"][k]
    lin, valid = load_gehler_image(name)
    est = sog_many(lin, valid, ps)
    row = {"image": name, "camera": d["cams"][k], "valid_ratio": round(float(valid.mean()), 3)}
    for m, p in methods.items():
        row.update({f"{m}_{ch}": float(est[p][j]) for j, ch in enumerate("rgb")})
        row[f"{m}_err"] = float(angular_error(est[p], d["illum"][k]))
    row["baseline_err"] = float(angular_error(base[d["cams"][k]], d["illum"][k]))
    rows.append(row)
    if (k + 1) % 20 == 0 or k + 1 == n:
        print(f"  {k + 1:3d}/{n} ({time.time() - t0:.0f}초)")

res = pd.DataFrame(rows)
res.to_csv(OUT / f"gehler_est_{split}.csv", index=False)

print("\n[각도 오차 중앙값 (도)]   전체 · 1D · 5D")
for m in ["baseline"] + list(methods):
    s = summarize(res[f"{m}_err"])
    m1, m5 = (res.loc[res.camera == cam, f"{m}_err"].median() for cam in ["1D", "5D"])
    print(f"  {m:15s} {s['median']:5.2f} · {m1:5.2f} · {m5:5.2f}   (나쁜 25% {s['worst25']:5.2f})")
n_fail = int(res[[f"{m}_r" for m in methods]].isna().any(axis=1).sum())
print(f"추정 실패(쓸 픽셀 없음) 사진: {n_fail}장")
if split == "calib":
    print("※ 확인: gray_world · shades_of_gray · white_patch 중앙값이 16 출력(3.72 · 2.51 · 6.93)과 같아야 정상")
print(f"\n저장: {(OUT / f'gehler_est_{split}.csv').relative_to(ROOT)}" + (", reports/illum_params.json (baseline)" if split == "calib" else ""))