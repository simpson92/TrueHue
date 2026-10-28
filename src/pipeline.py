# src/pipeline.py
# 목적: "카메라 RGB → 색 이름" 처리 흐름을 한 곳에 모은다
#   기준선 · 보정 3종 · 상한선 채점, 08(행렬 맞추기), 웹 화면이 모두 이 함수들을 같이 쓴다
#   → 방법마다 다른 것은 "조명 색" 하나뿐이고, 나머지 처리는 모두 똑같다 (D6)
#
#   카메라 RGB (블랙 레벨 뺀 값)
#     → ① ÷ 조명 이득 [r/g, 1, b/g]      apply_illuminant()   ← 방법마다 다른 유일한 단계
#     → ② × 노출 배율 (회색 칸 G 기준)    normalize_exposure()
#     → ③ 3×3 행렬                       to_xyz()
#     → ④ XYZ → Lab → 이름               to_lab(), name_patches()
import json
import numpy as np
import colour
from common import ROOT, NEUTRALS
from namer import name_color

# ── 컬러체커 공식 표준 색 (XYZ, D50 기준, 흰색 Y=1) ──────────────
_CC = colour.CCS_COLOURCHECKERS["ColorChecker24 - Before November 2014"]
XYZ_STD = colour.xyY_to_XYZ(np.array(list(_CC.data.values())))   # (24, 3)
D50 = _CC.illuminant
LAB_STD = colour.XYZ_to_Lab(XYZ_STD, D50)                         # (24, 3)


def apply_illuminant(patches, illum):
    """① 조명 색으로 나누기. G는 1로 고정해 '색만' 보정하고 밝기는 건드리지 않는다.

    patches: (n, 24, 3) 또는 (24, 3) / illum: (n, 3) 또는 (3,)
    """
    illum = np.asarray(illum, float)
    gains = illum / illum[..., 1:2]                       # [r/g, 1, b/g]
    return patches / gains[..., None, :]


def normalize_exposure(wb):
    """② 사진마다 다른 노출을 맞춘다: 회색 칸(20~23번)의 G 값이 표준 Y와 같아지는 배율 하나를 곱함.

    G 채널은 ①에서 바뀌지 않으므로, 이 배율은 어떤 조명 추정 방법을 써도 똑같다.
    """
    nidx = [k - 1 for k in NEUTRALS]
    g = wb[..., nidx, 1]                                  # (n, 4)
    y = XYZ_STD[nidx, 1]                                  # (4,)
    scale = np.nansum(g * y, axis=-1) / np.nansum(g * g, axis=-1)   # 사진별 최소제곱 배율
    return wb * scale[..., None, None]


def fit_matrix(rgb, xyz):
    """rgb(칸 수, 3) @ M.T ≈ xyz(칸 수, 3) 가 되는 3×3 행렬 M을 최소제곱으로 구한다 (08에서 사용)"""
    ok = ~np.isnan(rgb).any(axis=1)
    Mt, *_ = np.linalg.lstsq(rgb[ok], xyz[ok], rcond=None)
    return Mt.T, ok


def load_color_matrices():
    """08이 저장한 카메라별 행렬 → {"1D": M, "5D": M}"""
    data = json.loads((ROOT / "reports" / "color_matrix.json").read_text())
    return {cam: np.array(data[cam]["M"]) for cam in ["1D", "5D"]}


def to_xyz(rgb, cams, matrices):
    """③ 카메라별 행렬로 RGB → XYZ. rgb: (n, 24, 3), cams: (n,)"""
    xyz = np.full_like(rgb, np.nan)
    for cam, M in matrices.items():
        sel = cams == cam
        xyz[sel] = rgb[sel] @ M.T
    return xyz


def to_lab(xyz):
    """④-1 XYZ → Lab (D50 흰색 기준)"""
    return colour.XYZ_to_Lab(xyz, D50)


def name_patches(lab):
    """④-2 Lab (n, 24, 3) → 이름 (n, 24). 빈칸(NaN)은 None"""
    names = np.empty(lab.shape[:-1], dtype=object)
    for idx in np.ndindex(lab.shape[:-1]):
        v = lab[idx]
        names[idx] = None if np.isnan(v).any() else name_color(v)
    return names


def patches_to_lab(patches, illum, cams, matrices):
    """①~④-1 한 번에: 24칸 색 + 조명 색 → Lab. 기준선·보정·상한선은 illum만 바꿔 부른다"""
    wb = normalize_exposure(apply_illuminant(patches, illum))
    return to_lab(to_xyz(wb, cams, matrices))