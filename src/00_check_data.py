# src/00_check_data.py
# 목적: 데이터셋 PNG가 제대로 받아졌는지, "선형 16비트 이미지"가 맞는지 숫자로 확인
from pathlib import Path
import cv2

# 이 파일(src/) 기준 한 단계 위 = 프로젝트 루트(truehue/)
ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "gehler_shi": ROOT / "data" / "gehler_shi" / "png",   # 기대: 568장
    "cube_plus":  ROOT / "data" / "cube_plus",            # 기대: 1707장 (아직 안 받음)
}

for name, folder in DATASETS.items():
    pngs = sorted(folder.rglob("*.png"))    # 하위 폴더까지 .png 모두 찾기
    print(f"\n=== {name}: PNG {len(pngs)}장 ===")

    if not pngs:
        print("  (아직 데이터 없음)")
        continue

    # 카메라별로 하나씩 보이도록: 1D(IMG_로 시작)와 5D(8D5U로 시작) 샘플 고르기
    samples = [p for p in pngs if p.name.startswith("IMG")][:2] + \
              [p for p in pngs if p.name.startswith("8D5U")][:2]

    for p in samples:
        # ⚠️ IMREAD_UNCHANGED가 핵심!
        #    기본 imread는 16비트를 8비트(0~255)로 깎아서 읽어 버려요
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

        if img is None:                     # 파일이 깨졌거나 경로가 틀린 경우
            print(f"  {p.name}: 읽기 실패")
            continue

        print(f"  {p.name}: shape={img.shape}, dtype={img.dtype}, "
              f"min={img.min()}, max={img.max()}")