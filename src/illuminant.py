# src/illuminant.py
# 목적: 사진 한 장에서 "조명 색"을 추정하는 부품 (D11). 13 이후 스크립트와 채점 · 웹 화면이 같이 쓴다
#   세 방법은 사실 같은 식 하나에서 p만 다르다 (Shades of Gray 일반식)
#     조명 색(채널 c) = ( 픽셀 값의 p제곱 평균 ) ^ (1/p)
#       p = 1   → 그냥 평균              = Gray World   ("장면 전체 평균은 회색이다")
#       p = ∞   → 가장 큰 값             = White Patch  ("가장 밝은 곳은 원래 흰색이다")
#       그 사이 → 밝은 곳에 무게를 더 둔 평균 = Shades of Gray (p는 보정용 사진으로 정함, D11)
#   - 넣는 값: 블랙 레벨을 뺀 선형 RGB(lin)와 "계산에 써도 되는 픽셀" 표시(valid)
#     valid에서 포화 픽셀과 기준 물체(컬러체커 · 큐브) 자리는 이미 빠져 있어야 한다 (공정성 마스킹)
#   - 돌려주는 값: 조명 색 (R, G, B). 밝기(길이)는 의미 없고 방향(색)만 쓴다 → 각도 오차로 비교
import numpy as np

METHODS = {"gray_world": 1, "white_patch": np.inf}      # 이름 → p (Shades of Gray는 p를 정한 뒤 추가)


def shades_of_gray(lin, valid, p):
    """선형 RGB (H, W, 3) + valid (H, W) → 조명 색 (3,). 쓸 픽셀이 없으면 NaN"""
    px = lin[valid].astype(np.float64)                  # (픽셀 수, 3)
    if px.size == 0:
        return np.full(3, np.nan)
    if np.isinf(p):
        return px.max(axis=0)                           # White Patch: 채널별 최댓값 (D11)
    top = px.max()
    if top <= 0:
        return np.full(3, np.nan)
    # 큰 p에서 숫자가 너무 커지지 않게 0~1로 줄여서 계산한 뒤 되돌림 (결과는 같음)
    return top * np.mean((px / top) ** p, axis=0) ** (1 / p)


def gray_world(lin, valid):
    """Gray World: 쓸 수 있는 픽셀의 채널별 평균"""
    return shades_of_gray(lin, valid, 1)


def white_patch(lin, valid):
    """White Patch: 쓸 수 있는 픽셀의 채널별 최댓값 (포화 픽셀 · 기준 물체는 valid에서 이미 빠짐)"""
    return shades_of_gray(lin, valid, np.inf)


def summarize(errors):
    """각도 오차 목록 → 색 항상성 연구에서 쓰는 표준 요약 (도)"""
    e = np.sort(np.asarray(errors, float))
    e = e[~np.isnan(e)]
    q1, q2, q3 = np.percentile(e, [25, 50, 75])
    k = max(1, len(e) // 4)
    return {"n": len(e), "mean": e.mean(), "median": q2,
            "trimean": (q1 + 2 * q2 + q3) / 4,          # 중앙값 근처에 무게를 둔 평균
            "best25": e[:k].mean(), "worst25": e[-k:].mean(), "max": e.max()}