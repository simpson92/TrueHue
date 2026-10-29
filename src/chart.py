# src/chart.py
# 목적: Gehler 사진 속 컬러체커 위치를 찾는 "부품" (11 · 12가 불러다 씀, 직접 실행하지 않음)
#   - detect_chart(): 09의 3차 방식 그대로 (정답 색으로 찾기). 계산은 한 글자도 바꾸지 않았고,
#                     검출에 실패하면 에러로 멈추는 대신 None을 돌려주도록만 고쳤다 (최종 점검 B2)
#   - mask_box()    : 찾은 상자를 넉넉하게 넓힌 "가림 영역" (최종 점검 B1)
#   - to_display()  : 사람이 눈으로 확인하기 위한 그림 (정답 조명으로 색만 맞춤, 계산에는 안 씀 — D8)
import cv2
import numpy as np
import colour

CHROMATIC = list(range(2, 18))             # 3~18번 칸 (0부터 세면 2~17)
WINDOWS = [16, 24, 36, 54, 80, 120, 180, 270, 400, 540]   # 절반 크기 기준 창 크기 (원본은 2배)
MASK_PAD = 0.25                            # 가림 영역 여유: 상자 긴 변의 25%씩 사방으로


def to_display(lin, illum, scale):
    """선형 값 → 정답 조명으로 색 맞춤 → 밝기 맞춤 → sRGB 감마 (사람 눈 확인용 그림, RGB 0~1)"""
    gains = np.asarray(illum, float) / illum[1]
    return colour.cctf_encoding(np.clip(lin / gains * scale, 0, 1))


def detect_chart(lin, patches, tol=0.2, floor=20.0):
    """유채색 16칸과 비슷한 픽셀들이 가장 많이 모인 자리 → (상자 x1,y1,x2,y2 원본 좌표, 찾은 칸 비율, 창 크기)

    검출할 수 없으면 (None, 0.0, 0)을 돌려준다 (예: 유채색 칸이 모두 빈칸, 맞는 색 픽셀이 하나도 없음)
    """
    f = 2                                                 # 계산을 빠르게 하려고 절반 크기에서 찾음
    small = cv2.resize(lin.astype(np.float32), None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    ks = [k for k in CHROMATIC if not np.isnan(patches[k]).any()]   # 빈칸이 아닌 유채색 칸만
    if not ks:
        return None, 0.0, 0
    masks = []
    for k in ks:
        c = patches[k].astype(np.float32)
        # 채널마다 "몇 배 차이 나는지"(로그 비율)로 비교 → 밝기 차이에 덜 민감
        d = np.max(np.abs(np.log((small + floor) / (c + floor))), axis=2)
        masks.append((d < tol).astype(np.float32))

    best = (-1, None, None, None)                         # (모인 칸 수, 창, x, y)
    for w in WINDOWS:                                     # 작은 창부터 → 같은 수면 작은 창이 남음
        # 창이 컬러체커만 하다면 칸 하나는 창 폭의 약 1/7 → 그 넓이의 1/4 이상 있어야 "그 색이 있다"
        min_px = max(2.0, 0.25 * (w / 7) ** 2)
        count = sum((cv2.blur(m, (w, w)) * w * w >= min_px).astype(np.int32) for m in masks)
        y, x = np.unravel_index(np.argmax(count), count.shape)
        if count[y, x] > best[0]:                         # "더 많이" 모였을 때만 바꿈
            best = (count[y, x], w, x, y)
    n_found, w, cx, cy = best
    if n_found <= 0:
        return None, 0.0, 0

    # 창 주변에서 칸별로 맞는 픽셀들의 가운데 위치 → 그 점들을 감싸는 상자 + 넉넉한 여유
    h, wd = small.shape[:2]
    r = int(w * 0.75)
    y1, y2, x1, x2 = max(cy - r, 0), min(cy + r, h), max(cx - r, 0), min(cx + r, wd)
    pts = []
    for m in masks:
        ys, xs = np.nonzero(m[y1:y2, x1:x2])
        if len(ys):
            pts.append((np.median(xs) + x1, np.median(ys) + y1))
    if not pts:                                           # B2: 여기서 비면 09는 에러로 멈췄음
        return None, 0.0, 0
    pts = np.array(pts)
    pad = 0.4 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 4)   # 컬러체커가 삐져나오지 않게
    box = np.array([pts[:, 0].min() - pad, pts[:, 1].min() - pad,
                    pts[:, 0].max() + pad, pts[:, 1].max() + pad]) * f
    return box, n_found / len(ks), w * f


def mask_box(box, shape, pad=MASK_PAD):
    """검출 상자 → 가림 영역 (긴 변의 pad배씩 사방으로 넓히고 사진 밖은 자름). 정수 (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = box
    p = pad * max(x2 - x1, y2 - y1)
    H, W = shape[:2]
    return (int(max(x1 - p, 0)), int(max(y1 - p, 0)), int(min(x2 + p, W)), int(min(y2 + p, H)))