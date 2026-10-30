# src/hint.py
# 목적: 웹 화면의 "○○에 가까움" 표시를 만드는 부품 (D13, 웹-3)
#   - 이름 자체는 namer.name_color 그대로 (평가와 같은 규칙, 바꾸지 않음). 여기는 그 위에 얹는 "표시"만 만든다
#   - 규칙 경계(채도 10 · 색상 구간 경계 · 밝기 28/88/50) 가까이에 있으면, 경계 건너편 이름을 함께 알려 준다
#     예: 채도 8.5인 회색 → "회색 (파랑에 가까움)"
#   - 얼마나 가까워야 표시할지(기준 여유)는 22가 보정용 사진으로 정해 reports/web_params.json에 적는다
import numpy as np
from namer import name_color, lab_to_lch, CHROMA_GRAY, L_BLACK, L_WHITE, L_BROWN, HUE_SECTORS

KINDS = ["chroma", "hue", "L"]                     # 경계 종류: 채도 · 색상 각도 · 밝기
STEP = 0.01                                        # 경계를 "살짝 넘었을 때"의 거리


def _lab(L, C, h):
    r = np.radians(h)
    return (L, C * np.cos(r), C * np.sin(r))


def neighbors(lab):
    """Lab → [(경계 종류, 여유, 건너편 이름), ...]  (건너편 이름이 지금 이름과 다른 것만)

    여유는 namer.boundary_margin과 같은 단위: 채도 차 · 색상 각도 차(도) · 밝기 차
    """
    L, C, h = lab_to_lch(lab)
    base = name_color(lab)
    out = []
    # ① 채도 경계 (무채색 ↔ 유채색): 같은 밝기 · 같은 색상 각도에서 채도만 경계 건너편으로
    C2 = CHROMA_GRAY + STEP if C < CHROMA_GRAY else CHROMA_GRAY - STEP
    out.append(("chroma", abs(C - CHROMA_GRAY), name_color(_lab(L, C2, h))))
    if C >= CHROMA_GRAY:
        # ② 색상 경계: 가장 가까운 구간 경계를 살짝 넘긴 색상 각도
        edges = sorted({s for s, _, _ in HUE_SECTORS} | {360})
        diff = {e: ((h - e + 180) % 360) - 180 for e in edges}          # 부호 있는 차 (음수 = 경계보다 작음)
        e = min(edges, key=lambda e: abs(diff[e]))
        h2 = (e + STEP if diff[e] < 0 else e - STEP) % 360
        out.append(("hue", abs(diff[e]), name_color(_lab(L, C, h2))))
        # ③ 밝기 경계 (주황 ↔ 갈색)
        if 36 <= h < 82:
            L2 = L_BROWN + STEP if L < L_BROWN else L_BROWN - STEP
            out.append(("L", abs(L - L_BROWN), name_color(_lab(L2, C, h))))
    else:
        # ③ 밝기 경계 (검정 ↔ 회색 ↔ 흰색): 가까운 쪽
        edge = L_BLACK if abs(L - L_BLACK) < abs(L - L_WHITE) else L_WHITE
        L2 = edge + STEP if L < edge else edge - STEP
        out.append(("L", abs(L - edge), name_color(_lab(L2, C, h))))
    return [(k, float(m), n) for k, m, n in out if n != base]


def near_name(lab, thresholds):
    """Lab + 기준 여유 {"chroma": t, "hue": t, "L": t} → 함께 보여 줄 이웃 이름 (없으면 None)

    여러 경계가 기준 안이면, 기준 대비 가장 가까운 것(여유 / 기준이 가장 작은 것) 하나만
    """
    cands = [(m / thresholds[k], n) for k, m, n in neighbors(lab) if thresholds.get(k, 0) > 0 and m < thresholds[k]]
    return min(cands)[1] if cands else None


# ── 직접 실행하면: 자기 점검 ─────────────────────────────────────
if __name__ == "__main__":
    T = {"chroma": 3.0, "hue": 5.0, "L": 4.0}
    cases = [
        ((50, 0, -8.5), "회색", "파랑"),        # 채도 8.5, 색상 270° → 경계 건너편은 파랑
        ((50, 0, -12), "파랑", "회색"),         # 채도 12 → 건너편은 회색
        ((50, 20, 6.0), "빨강", "주황"),        # 색상 약 16.7°… 경계 36°까지 멀다 → 채도 · 색상 모두 기준 밖
        ((60, 25 * np.cos(np.radians(34)), 25 * np.sin(np.radians(34))), "빨강", "주황"),   # 색상 34° → 주황 경계 2°
        ((48, 30 * np.cos(np.radians(60)), 30 * np.sin(np.radians(60))), "갈색", "주황"),  # 밝기 48 → 50 경계 2
        ((30, 1, 1), "회색", "검정"),           # 밝기 30 → 검정 경계 28까지 2
        ((90, 1, 1), "흰색", "회색"),           # 밝기 90 → 흰색 경계 88까지 2
    ]
    for lab, want_name, want_near in cases:
        got = name_color(lab)
        near = near_name(lab, T)
        print(f"Lab {np.round(lab, 1)} → {got}" + (f" ({near}에 가까움)" if near else ""))
        assert got == want_name, (lab, got)
    assert near_name(cases[0][0], T) == "파랑"
    assert near_name(cases[1][0], T) == "회색"
    assert near_name(cases[2][0], T) is None
    assert near_name(cases[3][0], T) == "주황"
    assert near_name(cases[4][0], T) == "주황"
    assert near_name(cases[5][0], T) == "검정"
    assert near_name(cases[6][0], T) == "회색"
    # 기준 0이면 표시 없음
    assert near_name(cases[0][0], {"chroma": 0, "hue": 0, "L": 0}) is None
    print("✅ hint.py 부품 점검 통과")