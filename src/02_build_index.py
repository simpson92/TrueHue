# src/02_build_index.py
# 목적: 흩어진 정보(PNG 파일명 · 카메라 · 블랙 레벨 · 24칸 색 · 조명 정답)를
#       "사진 한 장 = 한 줄" 표로 묶고, 설명서 해석이 맞는지 검산한다
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
GS = ROOT / "data" / "gehler_shi"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ── 1. 파일 이름 정렬 & 카메라 구분 ─────────────────────────
names = sorted(p.name for p in (GS / "png").glob("*.png"))
# 8D5U로 시작하면 Canon 1D, IMG_로 시작하면 Canon 5D (shape로 확인한 사실)
cameras = np.array(["1D" if n.startswith("8D5U") else "5D" for n in names])

print(f"전체 {len(names)}장 | 1D {np.sum(cameras == '1D')}장 | 5D {np.sum(cameras == '5D')}장")
assert len(names) == 568, "PNG가 568장이 아니에요"
# 설명서: 1~86번이 1D → 정렬 후 앞 86장이 전부 1D여야 순서가 맞음
assert np.all(cameras[:86] == "1D") and np.all(cameras[86:] == "5D"), \
    "정렬 순서가 설명서(1D 86장 → 5D 482장)와 달라요"

# ── 2. 정답 .mat 불러오기 ───────────────────────────────────
patches = loadmat(GS / "colourchecker_gamma1_bit12.mat")["real_rgb"]   # (568, 24, 3)
illum_file = next(GS.glob("real_illum_568*.mat"))                     # 파일명에 점이 두 개라 glob 사용
illum_gt = loadmat(illum_file)["real_rgb"]                             # (568, 3)

# ── 3. 블랙 레벨 빼기 (+ 음수는 0으로 자르기) ───────────────
black = np.where(cameras == "1D", 0, 129)                  # 사진마다 빼야 할 값
# black[:, None, None] : (568,) → (568,1,1)로 모양을 늘려 (568,24,3)과 계산되게 함 (broadcasting)
patches_lin = np.clip(patches - black[:, None, None], 0, None)

# ── 4. 검산: 설명서 방식으로 정답 조명을 다시 계산해 보기 ─────
# 먼저 빈칸(NaN)이 얼마나, 어디에 있는지 확인
nan_mask = np.isnan(patches)                                  # (568,24,3) True = 빈칸
print(f"24칸 값 중 NaN: {nan_mask.sum()}개 "
      f"(NaN이 있는 사진 {nan_mask.any(axis=(1, 2)).sum()}장)")
# axis=(0,2): 사진·채널 방향으로 모아서 "몇 번 칸에 NaN이 있나" 확인 (+1: 1번부터 세기)
print(f"NaN이 나온 칸 번호: {np.where(nan_mask.any(axis=(0, 2)))[0] + 1}")
print(f"정답 조명 중 NaN: {np.isnan(illum_gt).sum()}개")

# MATLAB max처럼 NaN을 건너뛰려면 np.nanmax 사용
recomputed = np.nanmax((patches - black[:, None, None])[:, 18:24, :], axis=1)
diff = np.nanmax(np.abs(recomputed - illum_gt))
print(f"정답 조명 재계산 오차(최대): {diff:.6f}  ← 0에 가까우면 설명서 해석 정확!")

# ── 5. 한 줄 = 사진 한 장인 표 만들기 ───────────────────────
df = pd.DataFrame({"image": names, "camera": cameras, "black_level": black})
df[["illum_r", "illum_g", "illum_b"]] = illum_gt
for k in range(24):                                        # 24칸 × RGB = 72개 열
    for c, ch in enumerate("rgb"):
        df[f"p{k+1:02d}_{ch}"] = patches_lin[:, k, c]

df.to_csv(OUT / "gehler_index.csv", index=False)
print(f"저장 완료: outputs/gehler_index.csv  ({df.shape[0]}행 × {df.shape[1]}열)")
print(df.iloc[:3, :7])                                     # 앞 3줄, 앞 7열만 미리보기