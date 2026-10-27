# src/01_inspect_gt.py
# 목적: 정답 .mat 파일 안에 어떤 변수가, 어떤 모양(shape)으로 들어 있는지 확인
from pathlib import Path
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
GT_DIR = ROOT / "data" / "gehler_shi"

for mat_path in sorted(GT_DIR.glob("*.mat")):
    print(f"\n=== {mat_path.name} ===")
    data = loadmat(mat_path)              # .mat 파일을 파이썬 딕셔너리로 읽기

    for key, value in data.items():
        if key.startswith("__"):          # __header__ 같은 MATLAB 기본 정보는 건너뛰기
            continue
        # shape: 배열 모양 (예: (568, 3) = 사진 568장 × RGB 3개)
        print(f"  {key}: shape={getattr(value, 'shape', None)}, "
              f"dtype={getattr(value, 'dtype', None)}")