# src/namer.py
# 목적: CIELAB 색 하나를 받아 11색 중 하나의 이름을 돌려주는 "규칙 기반" 이름 붙이기
#   - 06 스크립트(검증), 이후 기준선·보정 채점, 웹 화면이 모두 이 함수를 같이 쓴다
#   - 규칙의 기준값은 데이터 사진을 전혀 보지 않고, 컬러체커 "공식 표준 색"만 보고 정했다
import numpy as np

# ── 규칙 기준값 (평가 전에 고정) ─────────────────────────────
CHROMA_GRAY = 10.0      # 채도가 이보다 낮으면 무채색 (표준 색: 회색 칸 ≤ 1.3, 유채색 칸 ≥ 19.5)
L_BLACK = 28.0          # 무채색 중 밝기가 이보다 낮으면 검정 (black 20.5 / neutral 3.5 35.7 사이)
L_WHITE = 88.0          # 무채색 중 밝기가 이보다 높으면 흰색 (neutral 8 81.3 / white 96.5 사이)
L_BROWN = 50.0          # 주황 계열 색상에서 밝기가 이보다 낮으면 갈색 (dark skin 38 / orange 62.7 사이)

# 색상 각도(hue, 도) 구간: (시작, 끝, 이름) — 시작 이상, 끝 미만
HUE_SECTORS = [
    (0,   36,  "빨강"),   # moderate red 18.6, red 27.8
    (36,  82,  "주황"),   # light skin 44.5, orange 57.7, orange yellow 74.1 (밝기 낮으면 갈색)
    (82,  100, "노랑"),   # yellow 87.1
    (100, 200, "초록"),   # yellow green 112.5, foliage 120.9, green 140.7, bluish green 180.3
    (200, 300, "파랑"),   # cyan 225, blue sky 257.4, purplish blue 282.8, blue 285.7, blue flower 289.2
    (300, 330, "보라"),   # purple 316.8
    (330, 360, "분홍"),   # magenta 343.7
]


def lab_to_lch(lab):
    """Lab → (L, 채도 C, 색상 각도 h[도])"""
    L, a, b = lab
    return L, float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)) % 360)


def name_color(lab):
    """CIELAB 색 (L, a, b) → 11색 이름 하나"""
    L, C, h = lab_to_lch(lab)

    # 1) 채도가 낮으면 무채색: 밝기로 검정 / 회색 / 흰색
    if C < CHROMA_GRAY:
        if L < L_BLACK:
            return "검정"
        if L > L_WHITE:
            return "흰색"
        return "회색"

    # 2) 유채색: 색상 각도로 구간 찾기
    for start, end, name in HUE_SECTORS:
        if start <= h < end:
            # 주황 계열인데 어두우면 갈색
            if name == "주황" and L < L_BROWN:
                return "갈색"
            return name
    raise ValueError(f"색상 각도 {h}가 어느 구간에도 없음")   # 구간이 0~360을 빈틈없이 덮으므로 오지 않음


def boundary_margin(lab):
    """가장 가까운 규칙 경계까지의 여유 (작을수록 조금만 색이 틀어져도 이름이 바뀜)"""
    L, C, h = lab_to_lch(lab)
    if C < CHROMA_GRAY:                               # 무채색: 채도 경계와 밝기 경계까지 거리
        return {"chroma": CHROMA_GRAY - C,
                "L": min(abs(L - L_BLACK), abs(L - L_WHITE))}
    edges = sorted({s for s, _, _ in HUE_SECTORS} | {360})
    d_hue = min(min(abs(h - e), 360 - abs(h - e)) for e in edges)
    out = {"chroma": C - CHROMA_GRAY, "hue": d_hue}
    if 36 <= h < 82:                                  # 주황/갈색 구간이면 밝기 경계도
        out["L"] = abs(L - L_BROWN)
    return out