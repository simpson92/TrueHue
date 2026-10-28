# src/03_inspect_nan.py
# 목적: 24칸 색 데이터의 빈칸(NaN)이 "어떤 패턴"으로 생겼는지 숫자로 조사한다
#       → 채점 가능한 문제 수 확정 + 빈칸이 생긴 이유 추리
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
GS = ROOT / "data" / "gehler_shi"

names = sorted(p.name for p in (GS / "png").glob("*.png"))
cameras = np.array(["1D" if n.startswith("8D5U") else "5D" for n in names])
patches = loadmat(GS / "colourchecker_gamma1_bit12.mat")["real_rgb"]     # (568, 24, 3)

# ── 1. 빈칸은 "칸 단위"인가, "채널 단위"인가? ─────────────────
any_nan = np.isnan(patches).any(axis=2)      # (568, 24) RGB 중 하나라도 빈칸
all_nan = np.isnan(patches).all(axis=2)      # (568, 24) RGB 셋 다 빈칸
print(f"빈칸이 있는 칸: {any_nan.sum()}  |  RGB 셋 다 빈칸인 칸: {all_nan.sum()}")
print("→ 두 숫자가 같으면 '칸 통째로' 비워 둔 것\n")

# ── 2. 채점 가능한 문제 수 ──────────────────────────────────
usable = (~any_nan).sum()
print(f"채점 가능한 문제: {usable} / {any_nan.size} ({usable / any_nan.size:.1%})\n")

# ── 3. 사진 한 장에 빈칸이 몇 개씩? ─────────────────────────
per_image = any_nan.sum(axis=1)              # 사진마다 빈칸 개수 (0~24)
# bincount: 빈칸 0개인 사진 몇 장, 1개인 사진 몇 장, ... 24개인 사진 몇 장
counts = np.bincount(per_image, minlength=25)
print("사진당 빈칸 개수 → 사진 수")
for k, n in enumerate(counts):
    if n > 0:
        print(f"  빈칸 {k:2d}개: {n}장")

# ── 4. 카메라별 차이 ────────────────────────────────────────
for cam in ["1D", "5D"]:
    sel = cameras == cam
    print(f"\n[{cam}] 빈칸 있는 사진 {np.sum(per_image[sel] > 0)} / {sel.sum()}장")

# ── 5. 칸 번호별 빈칸 횟수 ──────────────────────────────────
print("\n칸 번호별 빈칸 사진 수")
per_patch = any_nan.sum(axis=0)
for k, n in enumerate(per_patch, start=1):
    print(f"  {k:2d}번 칸: {n:3d}장 " + "█" * (n // 5))   # 막대로 대충 모양 보기

# ── 6. 가설 확인: 남은(빈칸 아닌) 칸의 "가장 어두운 채널" 값 ──
black = np.where(cameras == "1D", 0, 129)
lin = patches - black[:, None, None]       # 블랙 레벨 뺀 선형 값
min_ch = lin.min(axis=2)                   # (568, 24) 칸마다 가장 어두운 채널 (빈칸은 NaN)

for cam in ["1D", "5D"]:
    sel = (cameras == cam)[:, None] & ~any_nan        # 이 카메라의 남은 칸만
    vals = min_ch[sel]
    print(f"\n[{cam}] 남은 칸의 가장 어두운 채널 → "
          f"최솟값 {vals.min():.1f}, 하위 1% {np.percentile(vals, 1):.1f}")