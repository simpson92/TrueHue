# src/07_make_splits.py
# 목적: 사진을 "보정용(calib)"과 "봉인 평가용(test)"으로 나누고 목록을 splits/에 저장한다
#   - calib: 카메라 색 변환·설정값을 정할 때만 사용
#   - test : 마지막에 단 한 번 점수를 매길 때만 사용 (그 전에는 열지 않음)
#   - 한 번 만든 목록은 커밋해서 고정하고, 다시 만들지 않는다
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"
SPLITS.mkdir(exist_ok=True)
SEED = 42              # 무작위 시작값 고정 → 누가 돌려도 같은 결과
CALIB_RATIO = 1 / 3    # 보정용 1/3, 평가용 2/3

# ── 1. Gehler-Shi: 카메라별로 같은 비율로 나누기 ─────────────
names = sorted(p.name for p in (ROOT / "data" / "gehler_shi" / "png").glob("*.png"))
cameras = np.array(["1D" if n.startswith("8D5U") else "5D" for n in names])
assert len(names) == 568

rng = np.random.default_rng(SEED)
calib, test = [], []
for cam in ["1D", "5D"]:
    cam_names = [n for n, c in zip(names, cameras) if c == cam]
    shuffled = rng.permutation(cam_names)               # 이 카메라 사진만 무작위로 섞기
    n_calib = round(len(shuffled) * CALIB_RATIO)
    calib += sorted(shuffled[:n_calib])
    test += sorted(shuffled[n_calib:])
    print(f"[{cam}] 전체 {len(cam_names)} → 보정용 {n_calib} / 평가용 {len(cam_names) - n_calib}")

assert not set(calib) & set(test), "보정용과 평가용이 겹침"      # 한 사진이 양쪽에 있으면 안 됨
assert len(calib) + len(test) == 568

(SPLITS / "gehler_calib.txt").write_text("\n".join(calib) + "\n")
(SPLITS / "gehler_test.txt").write_text("\n".join(test) + "\n")
print(f"Gehler-Shi: 보정용 {len(calib)}장 / 평가용 {len(test)}장 저장")

# 채점 가능한 칸(빈칸 제외)이 양쪽에 얼마나 있는지도 확인
index = pd.read_csv(ROOT / "outputs" / "gehler_index.csv")       # 02 결과
patch_cols = [f"p{k:02d}_r" for k in range(1, 25)]
usable = index.set_index("image")[patch_cols].notna().sum(axis=1)  # 사진마다 채점 가능한 칸 수
print(f"  채점 가능 문제: 보정용 {usable[calib].sum()} / 평가용 {usable[test].sum()}")

# ── 2. SimpleCube++: 공식 분할 그대로 목록만 저장 ─────────────
SC = ROOT / "data" / "simplecube" / "SimpleCube++"
for split, out in [("train", "simplecube_calib.txt"), ("test", "simplecube_test.txt")]:
    ids = sorted(p.stem for p in (SC / split / "PNG").glob("*.png"))
    (SPLITS / out).write_text("\n".join(ids) + "\n")
    print(f"SimpleCube++ {split} → {out}: {len(ids)}장")