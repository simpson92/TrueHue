# src/05_check_simplecube.py
# 목적: SimpleCube++가 제대로 받아졌는지, 설명서에 없는 정보(블랙 레벨 등)를 숫자로 확인
#   1) train/test 사진 장수   2) 정답(gt.csv)과 사진이 1:1로 짝지어지는지
#   3) 라벨(properties) 종류  4) PNG 자료형·값 범위 (블랙 레벨·포화 추정)
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "data" / "simplecube" / "SimpleCube++"

for split in ["train", "test"]:
    print(f"\n{'=' * 15} {split} {'=' * 15}")

    # ── 1. 사진 장수 ────────────────────────────────────────
    pngs = sorted((SC / split / "PNG").glob("*.png"))
    print(f"PNG: {len(pngs)}장")

    # ── 2. 정답 파일과 짝 맞추기 ────────────────────────────
    gt = pd.read_csv(SC / split / "gt.csv")
    print(f"gt.csv: {len(gt)}줄, 열 = {list(gt.columns)}")
    print(gt.head(3).to_string(index=False))

    # 정답 파일의 image 열과 PNG 파일 이름(확장자 뺀 것)을 집합으로 비교
    # ※ '순서'가 아니라 '이름'으로 짝지으므로 정렬 문제가 생기지 않음
    gt_ids = set(gt["image"].astype(str).str.replace(".png", "", regex=False))
    png_ids = {p.stem for p in pngs}
    print(f"이름이 서로 맞는 사진: {len(gt_ids & png_ids)}장 | "
          f"정답만 있음: {len(gt_ids - png_ids)} | 사진만 있음: {len(png_ids - gt_ids)}")

    # 조명 정답 열 이름은 짐작하지 않고, 'image'를 뺀 나머지 숫자 열을 사용
    rgb_cols = [c for c in gt.columns if c != "image"]
    print(f"조명 정답 열: {rgb_cols}")
    rgb_sum = gt[rgb_cols].sum(axis=1)
    print(f"세 값의 합: 최소 {rgb_sum.min():.4f}, 최대 {rgb_sum.max():.4f}  ← 둘 다 1이면 정상")

    # ── 3. 라벨(properties) 확인 ────────────────────────────
    prop = pd.read_csv(SC / "auxiliary" / f"{split}_properties.csv")
    print(f"\nproperties: {len(prop)}줄, 열 = {list(prop.columns)}")
    # 평가에 쓸 만한 열들의 값 종류와 개수 보기 (없는 열은 건너뜀)
    for col in ["ds_version", "daytime", "place", "illumination", "is_sharp", "shadows"]:
        if col in prop.columns:
            print(f"  [{col}] " + ", ".join(f"{k}: {v}" for k, v in
                                           prop[col].value_counts(dropna=False).items()))

    # ── 4. PNG 값 범위 (앞 5장 샘플) ────────────────────────
    print()
    mins, maxs = [], []
    for p in pngs[:5]:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)   # ⚠️ 16비트 그대로 읽기
        mins.append(int(img.min())); maxs.append(int(img.max()))
        print(f"  {p.name}: shape={img.shape}, dtype={img.dtype}, "
              f"min={img.min()}, max={img.max()}")
    print(f"  → 샘플 최솟값들 {mins}  (모두 비슷하면 그게 블랙 레벨 후보)")
    print(f"  → 샘플 최댓값들 {maxs}  (같은 값에서 자주 멈추면 그게 포화 기준 후보)")


# ── 5. 블랙 레벨·포화 확인 (train 앞 50장) ──────────────────
print(f"\n{'=' * 15} 블랙 레벨 · 포화 확인 {'=' * 15}")
pngs = sorted((SC / "train" / "PNG").glob("*.png"))[:50]
nonzero_mins, maxs, zero_ratios = [], [], []

for p in pngs:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    zero = np.all(img == 0, axis=2)               # 세 채널이 모두 0인 픽셀 = 칠해진 자리 후보
    zero_ratios.append(zero.mean())
    nonzero_mins.append(int(img[~zero].min()))    # 칠해진 자리를 뺀 진짜 최솟값
    maxs.append(int(img.max()))

print(f"0인 픽셀 비율: 평균 {np.mean(zero_ratios):.1%}  (큐브 자리라면 몇 % 정도로 비슷해야 함)")
print(f"0이 아닌 픽셀의 최솟값: 가장 작은 값 {min(nonzero_mins)}, 중앙값 {int(np.median(nonzero_mins))}")
print(f"  → 2048 근처면 블랙 레벨이 그대로 남아 있는 것 (빼야 함)")
print(f"최댓값 상위 빈도: {pd.Series(maxs).value_counts().head(5).to_dict()}")
print(f"  → 가장 자주 나오는 값이 포화 기준")

# 0인 영역이 사진 어디에 있는지 (첫 장 기준)
img = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
ys, xs = np.where(np.all(img == 0, axis=2))
if len(ys):
    print(f"첫 장의 0 영역: 행 {ys.min()}~{ys.max()}, 열 {xs.min()}~{xs.max()} "
          f"(사진 크기 {img.shape[0]}×{img.shape[1]})") 