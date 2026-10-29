# src/common.py
# 목적: 여러 스크립트가 같이 쓰는 "공통 부품" 모음
#   - 데이터 경로와 처리 규칙 숫자(블랙 레벨, 포화 기준 등)를 한 곳에서 관리
#   - 사진은 반드시 read_rgb()로 읽는다  ← OpenCV의 BGR 순서를 RGB로 바꿔 주는 유일한 입구
# 직접 실행하면(python src/common.py) 채널 순서가 맞는지 스스로 점검한다
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# ── 경로 ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
GEHLER_DIR = ROOT / "data" / "gehler_shi"
SC_DIR = ROOT / "data" / "simplecube" / "SimpleCube++"

# ── 처리 규칙 숫자 (데이터 노트에 근거 기록) ────────────────────
GEHLER_BLACK = {"1D": 0, "5D": 129}    # 카메라별 블랙 레벨
GEHLER_SAT_RAW = 3300                   # 원본 값이 이보다 크면 포화 의심 (SFU 기준)
SC_BLACK = 2048                         # SimpleCube++ 블랙 레벨
SC_SAT = 15000                          # SimpleCube++ 포화 기준 (원본 값)
NEUTRALS = [20, 21, 22, 23]             # 노출 맞추기에 쓰는 회색 칸


def gehler_camera(name):
    """Gehler 파일 이름 → 카메라 ('8D5U…' = 1D, 'IMG_…' = 5D, 00·08에서 사진 크기로 확인)"""
    return "1D" if name.startswith("8D5U") else "5D"


def read_rgb(path):
    """16비트 PNG를 '원래 값 그대로', 'R, G, B 순서'로 읽는다.

    ⚠️ cv2.imread는 컬러 사진을 B, G, R 순서로 돌려준다.
       24칸 색(.mat)과 조명 정답(gt.csv)은 R, G, B 순서이므로, 읽자마자 순서를 뒤집어 맞춘다.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)     # 16비트 그대로 (8비트로 깎지 않음)
    if img is None:
        raise FileNotFoundError(f"사진을 읽을 수 없음: {path}")
    return img[:, :, ::-1].copy()                         # BGR → RGB (마지막 축 순서 뒤집기)


def angular_error(est, gt):
    """두 조명 색 사이의 각도 오차(도). 밝기(길이)는 무시하고 색의 '방향'만 비교한다."""
    est, gt = np.asarray(est, float), np.asarray(gt, float)
    cos = np.sum(est * gt, axis=-1) / (np.linalg.norm(est, axis=-1) * np.linalg.norm(gt, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def load_simplecube(img_id, split="train"):
    """SimpleCube++ 사진 → (블랙 레벨 뺀 RGB 실수 배열, 계산에 써도 되는 픽셀 표시)

    처리 순서가 중요하다 (데이터 노트 참고):
      ① 원본에서 세 채널이 모두 0인 픽셀(큐브를 칠한 자리)을 먼저 표시
      ② 원본 값이 포화 기준 이상인 픽셀 표시
      ③ 그다음 블랙 레벨을 빼고 음수는 0으로
    """
    raw = read_rgb(SC_DIR / split / "PNG" / f"{img_id}.png").astype(np.float64)
    cube = np.all(raw == 0, axis=2)                       # ① 큐브 자리
    sat = np.any(raw >= SC_SAT, axis=2)                   # ② 포화
    lin = np.clip(raw - SC_BLACK, 0, None)                # ③ 블랙 레벨 빼기
    valid = ~cube & ~sat
    return lin, valid


def load_simplecube_gt(split="train"):
    """SimpleCube++ 조명 정답 → {사진 id: (r, g, b)}"""
    gt = pd.read_csv(SC_DIR / split / "gt.csv")
    cols = [c for c in gt.columns if c != "image"]        # mean_r, mean_g, mean_b
    return {str(k): v for k, v in zip(gt["image"].astype(str), gt[cols].to_numpy())}


_CHART_BOXES = None


def load_gehler_image(name):
    """Gehler 사진 → (블랙 레벨 뺀 RGB 실수 배열, 계산에 써도 되는 픽셀 표시)  ← 조명 추정용

    SimpleCube++의 load_simplecube와 같은 규칙 (D3 · D8):
      ① 원본 값이 포화 기준(3,300)을 넘는 채널이 하나라도 있는 픽셀 표시
      ② 컬러체커 가림 영역(reports/chart_boxes.csv의 x1, y1, x2, y2) 표시 → 공정성 마스킹
      ③ 블랙 레벨을 빼고 음수는 0으로
    가림 영역이 없는 사진은 멈춘다 (컬러체커가 안 가려진 채 조명을 추정하면 안 되므로)
    """
    global _CHART_BOXES
    if _CHART_BOXES is None:
        _CHART_BOXES = pd.read_csv(ROOT / "reports" / "chart_boxes.csv").set_index("image")
    raw = read_rgb(GEHLER_DIR / "png" / name).astype(np.float64)
    sat = np.any(raw > GEHLER_SAT_RAW, axis=2)            # ① 포화
    box = _CHART_BOXES.loc[name, ["x1", "y1", "x2", "y2"]]
    if box.isna().any():
        raise ValueError(f"{name}: 컬러체커 가림 영역이 없음 (reports/chart_boxes.csv 확인)")
    x1, y1, x2, y2 = box.astype(int)
    chart = np.zeros(raw.shape[:2], bool)
    chart[y1:y2, x1:x2] = True                            # ② 가림 영역 (세로 사진도 그대로 잘림)
    lin = np.clip(raw - GEHLER_BLACK[gehler_camera(name)], 0, None)   # ③ 블랙 레벨
    valid = ~sat & ~chart
    return lin, valid


def load_gehler_index(split=None):
    """02가 만든 outputs/gehler_index.csv → 사진별 배열 묶음 (dict)

    split: None(전체) · "calib"(보정용) · "test"(평가용 — 최종 평가 때만!)
    돌려주는 것:
      names (n,)  · cams (n,)  · black (n,)
      patches (n, 24, 3)  블랙 레벨 뺀 24칸 색. 빈칸과 포화 의심 칸은 NaN
      illum (n, 3)        정답 조명 (R, G, B)
    """
    index = pd.read_csv(ROOT / "outputs" / "gehler_index.csv").set_index("image")
    if split is not None:
        names = (ROOT / "splits" / f"gehler_{split}.txt").read_text().split()
        index = index.loc[names]
    cols = [f"p{k:02d}_{c}" for k in range(1, 25) for c in "rgb"]
    patches = index[cols].to_numpy().reshape(-1, 24, 3).copy()
    black = index["black_level"].to_numpy()
    # 포화 의심 칸 제외: 원본 값(= 뺀 값 + 블랙 레벨)이 기준보다 크면 NaN
    sat = ((patches + black[:, None, None]) > GEHLER_SAT_RAW).any(axis=2)
    patches[sat] = np.nan
    return {
        "names": index.index.to_numpy(),
        "cams": index["camera"].to_numpy(),
        "black": black,
        "patches": patches,
        "illum": index[["illum_r", "illum_g", "illum_b"]].to_numpy(),
        "n_saturated": int(sat.sum()),
    }


# ── 직접 실행하면: 채널 순서 점검 ─────────────────────────────
if __name__ == "__main__":
    # 원리: Gray World(사진 평균 색 = 조명 색)로 조명을 추정해 정답과 비교한다.
    #       채널 순서가 맞으면 각도 오차가 작고, R·B가 뒤바뀌면 크게 나와야 한다.
    gts = load_simplecube_gt("train")
    ids = sorted(gts)[:50]                                # train 앞 50장만 (test는 열지 않음)
    err_rgb, err_bgr = [], []
    for i in ids:
        lin, valid = load_simplecube(i, "train")
        est = lin[valid].mean(axis=0)                     # Gray World 추정 (R, G, B)
        err_rgb.append(angular_error(est, gts[i]))        # read_rgb 순서 그대로
        err_bgr.append(angular_error(est[::-1], gts[i]))  # 일부러 R·B를 뒤집으면?
    print(f"train {len(ids)}장, Gray World 각도 오차 중앙값")
    print(f"  RGB 순서(read_rgb): {np.median(err_rgb):5.2f}도")
    print(f"  R·B 뒤바뀜(BGR)   : {np.median(err_bgr):5.2f}도")
    assert np.median(err_rgb) < np.median(err_bgr), "채널 순서가 의심스러움"
    print("✅ read_rgb의 채널 순서가 정답 파일(R, G, B)과 맞음")