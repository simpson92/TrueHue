# src/04_reference_colors.py
# 목적: 컬러체커 24칸의 "공식 표준 색"을 CIELAB으로 꺼내고,
#       11색 기준색(xkcd 색 설문)과 비교해 정답표 초안을 만든다
#   - 결과 1: reports/patch_reference.csv  (칸별 공식 Lab · 화면용 색 · 가까운 이름 2개)
#   - 결과 2: outputs/patches_preview.png  (24칸을 눈으로 보며 이름을 정하기 위한 그림)
from pathlib import Path
import numpy as np
import pandas as pd
import colour                      # pip install colour-science (색 과학 계산 라이브러리)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

# ── 1. 컬러체커 24칸 공식 값 불러오기 ───────────────────────
# Gehler 데이터셋은 2008년 촬영 → 2014년 11월 이전 판 컬러체커 값을 사용
cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - Before November 2014"]
patch_names = list(cc.data.keys())                 # 'dark skin', 'light skin', ...
xyY = np.array(list(cc.data.values()))             # (24, 3) 공식 측정값 (D50 조명 기준)
lab = colour.XYZ_to_Lab(colour.xyY_to_XYZ(xyY), cc.illuminant)   # (24, 3) L*, a*, b*

# 화면에 보여 줄 sRGB 색 (그림용으로만 사용, 0~1 범위로 자름)
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
xyz_d65 = colour.chromatic_adaptation(colour.xyY_to_XYZ(xyY),
                                      colour.xy_to_XYZ(cc.illuminant), colour.xy_to_XYZ(D65))
srgb = np.clip(colour.XYZ_to_sRGB(xyz_d65), 0, 1)

# ── 2. 11색 기준색 (xkcd 색 설문의 대표색) ──────────────────
# 각 이름을 "가장 그 이름다운 색" 하나로 대표한 값 → 정답표를 정할 때 참고용
PROTOTYPES = {
    "빨강": "#e50000", "주황": "#f97306", "노랑": "#ffff14", "초록": "#15b01a",
    "파랑": "#0343df", "보라": "#7e1e9c", "분홍": "#ff81c0", "갈색": "#653700",
    "검정": "#000000", "흰색": "#ffffff", "회색": "#929591",
}

def hex_to_lab(h):
    """'#rrggbb' → sRGB(0~1) → XYZ → Lab (D65 흰색 기준)"""
    rgb = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb), D65)

proto_names = list(PROTOTYPES)
proto_lab = np.array([hex_to_lab(h) for h in PROTOTYPES.values()])   # (11, 3)

# ── 3. 칸마다 가장 가까운 기준색 2개 찾기 (ΔE2000 = 사람 눈 기준 색 차이) ──
rows = []
for i in range(24):
    d = np.array([colour.delta_E(lab[i], p, method="CIE 2000") for p in proto_lab])
    first, second = np.argsort(d)[:2]               # 가장 가까운 것, 두 번째로 가까운 것
    L, a, b = lab[i]
    rows.append({
        "patch": i + 1,
        "patch_name": patch_names[i],
        "L": round(L, 2), "a": round(a, 2), "b": round(b, 2),
        "chroma": round(np.hypot(a, b), 1),                 # 채도: 회색에서 얼마나 먼가
        "hue_deg": round(np.degrees(np.arctan2(b, a)) % 360, 1),   # 색상 각도
        "hex": "#" + "".join(f"{round(v * 255):02x}" for v in srgb[i]),
        "nearest_1": proto_names[first],  "dE_1": round(d[first], 1),
        "nearest_2": proto_names[second], "dE_2": round(d[second], 1),
        "margin": round(d[second] - d[first], 1),   # 작을수록 두 이름 사이에서 애매
    })

df = pd.DataFrame(rows)
df.to_csv(ROOT / "reports" / "patch_reference.csv", index=False)
print(df[["patch", "patch_name", "L", "chroma", "hue_deg",
          "nearest_1", "nearest_2", "margin"]].to_string(index=False))

# ── 4. 눈으로 확인할 그림 (6열 × 4행, 실제 컬러체커 배치) ──
fig, axes = plt.subplots(4, 6, figsize=(12, 8.4))
for i, ax in enumerate(axes.flat):
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=srgb[i]))
    text_color = "white" if lab[i][0] < 55 else "black"     # 어두운 칸엔 흰 글씨
    ax.text(0.5, 0.62, f"{i + 1}", ha="center", fontsize=16, weight="bold", color=text_color)
    ax.text(0.5, 0.38, patch_names[i].split(" (")[0], ha="center", fontsize=9, color=text_color)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "patches_preview.png", dpi=120)
print("\n저장: reports/patch_reference.csv, outputs/patches_preview.png")