# src/06_test_namer.py
# 목적: 이름 붙이기 규칙(namer.py)에 컬러체커 "공식 표준 색" 24개를 넣었을 때
#       정답표(answer_table.csv)대로 모두 맞히는지 확인한다  ← 이후 채점의 전제 조건
#   - 결과 1: 화면에 칸별 결과 + 경계까지의 여유
#   - 결과 2: outputs/namer_hue_chroma.png  (색상 각도·채도 평면에 24칸과 규칙 경계)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from namer import name_color, boundary_margin, HUE_SECTORS, CHROMA_GRAY

ROOT = Path(__file__).resolve().parents[1]

ref = pd.read_csv(ROOT / "reports" / "patch_reference.csv")    # 04 결과: 공식 Lab
ans = pd.read_csv(ROOT / "reports" / "answer_table.csv")       # 정답표 (초안)
df = ref.merge(ans[["patch", "answer", "answer_alt", "confidence"]], on="patch")   # 칸 번호로 합치기

rows = []
for _, r in df.iterrows():
    lab = (r["L"], r["a"], r["b"])
    pred = name_color(lab)
    # 정답 또는 추가 정답(애매한 칸)과 같으면 맞음
    accepted = {r["answer"]} | ({r["answer_alt"]} if isinstance(r["answer_alt"], str) else set())
    m = boundary_margin(lab)
    rows.append({
        "patch": r["patch"], "name": r["patch_name"].split(" (")[0],
        "정답": "/".join(sorted(accepted)), "규칙 결과": pred,
        "맞음": "O" if pred in accepted else "X",
        "색상 여유(도)": round(m["hue"], 1) if "hue" in m else "-",
        "채도 여유": round(m["chroma"], 1),
        "밝기 여유": round(m["L"], 1) if "L" in m else "-",
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False))
n_ok = (res["맞음"] == "O").sum()
print(f"\n표준 색 24개 중 {n_ok}개 정답")
assert n_ok == 24, "표준 색을 모두 맞히지 못함 → 규칙 기준값을 다시 정해야 함"
print("✅ 전제 조건 통과: 이후 채점에서 틀리는 것은 조명(과 카메라) 때문")

# ── 그림: 색상 각도·채도 평면 (극좌표) ─────────────────────────
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection="polar")
for start, _, _ in HUE_SECTORS:                         # 색상 경계선
    ax.plot([np.radians(start)] * 2, [CHROMA_GRAY, 85], color="gray", lw=0.8, ls="--")
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(th, [CHROMA_GRAY] * 200, color="black", lw=1)  # 무채색 경계 (채도 10)
for _, r in df.iterrows():
    ax.scatter(np.radians(r["hue_deg"]), r["chroma"], s=160, color=r["hex"],
               edgecolors="black", linewidths=0.8, zorder=3)
    ax.text(np.radians(r["hue_deg"]), r["chroma"] + 6, str(r["patch"]), ha="center", fontsize=8)
ax.set_rmax(85)
ax.set_title("ColorChecker 24 patches: hue angle x chroma (dashed = name boundaries)", fontsize=10)
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "namer_hue_chroma.png", dpi=120)
print("저장: outputs/namer_hue_chroma.png")