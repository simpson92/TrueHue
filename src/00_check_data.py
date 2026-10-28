# src/00_check_data.py
# 목적: 데이터셋 PNG가 제대로 받아졌는지, "선형 16비트 이미지"가 맞는지 숫자로 확인
#   - Gehler-Shi: 568장 / SimpleCube++: train 1,772장 + test 462장
#   (값의 범위만 보므로 채널 순서는 상관없음. 색을 쓰는 스크립트는 common.read_rgb 사용)
from pathlib import Path
import cv2

# 이 파일(src/) 기준 한 단계 위 = 프로젝트 루트
ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "gehler_shi":       (ROOT / "data" / "gehler_shi" / "png", 568),
    "simplecube_train": (ROOT / "data" / "simplecube" / "SimpleCube++" / "train" / "PNG", 1772),
    "simplecube_test":  (ROOT / "data" / "simplecube" / "SimpleCube++" / "test" / "PNG", 462),
}

for name, (folder, expected) in DATASETS.items():
    pngs = sorted(folder.glob("*.png"))
    ok = "✅" if len(pngs) == expected else "⚠️"
    print(f"\n=== {name}: PNG {len(pngs)}장 (기대 {expected}장) {ok} ===")

    if not pngs:
        print("  (데이터 없음 — README의 '데이터 준비' 참고)")
        continue

    if name == "gehler_shi":
        # 카메라별로 2장씩: 8D5U… = Canon 1D, IMG_… = Canon 5D (사진 크기로 확인한 사실)
        samples = [p for p in pngs if p.name.startswith("8D5U")][:2] + \
                  [p for p in pngs if p.name.startswith("IMG")][:2]
    else:
        samples = pngs[:2]

    for p in samples:
        # ⚠️ IMREAD_UNCHANGED가 핵심! 기본 imread는 16비트를 8비트(0~255)로 깎아서 읽어 버림
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:                     # 파일이 깨졌거나 경로가 틀린 경우
            print(f"  {p.name}: 읽기 실패")
            continue
        print(f"  {p.name}: shape={img.shape}, dtype={img.dtype}, "
              f"min={img.min()}, max={img.max()}")