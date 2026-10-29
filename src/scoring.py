# src/scoring.py
# 목적: 색 이름 "채점"과 "보고"에 쓰는 부품 (D10). 보정용 연습 채점(19)과 최종 1회 평가가 같은 함수를 쓴다
#   → 보정용에서 확인한 계산이 평가용에도 그대로 쓰인다는 보장
#   - 정답 판정: 규칙의 답이 {answer, answer_alt} 안에 있으면 정답 (D10 결정 1)
#   - 이름 일관성: 칸마다 기준 이름 = name_color(공식 표준 색)과 같은 비율 (D10 결정 1)
#   - 조명 그룹: 정답 조명 → 색좌표 xy → 색온도(McCamy) → 3그룹 (D10 결정 2, xy 계산은 17이 정한 행렬)
#   - 오차 범위: 사진 단위 부트스트랩, 카메라별로 따로 뽑기, 짝지은 비교 (D10 결정 3)
#   - 최종 평가 잠금: 코드 · 설정이 모두 커밋된 상태에서, 한 번만 (D5)
import json
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import colour
from common import ROOT
from namer import name_color
from pipeline import LAB_STD, load_color_matrices

# ── 정답표 · 기준 이름 ─────────────────────────────────────────
_ANS = pd.read_csv(ROOT / "reports" / "answer_table.csv").sort_values("patch")
ALLOWED = [{a} | ({b} if isinstance(b, str) else set())                  # 칸별 정답 묶음 (24개)
           for a, b in zip(_ANS["answer"], _ANS["answer_alt"])]
REF_NAMES = [name_color(LAB_STD[k]) for k in range(24)]                   # 일관성 기준 이름 (5번만 "파랑")

# 보고 범위 (D10 결정 1). 칸 번호는 1부터
SUBSETS = {
    "전체 24칸": list(range(1, 25)),
    "확실 16칸": [1, 3, 4, 7, 8, 10, 13, 14, 15, 16, 19, 21, 22, 9, 23, 24],
    "빨강 9·15": [9, 15],
    "초록 4·11·14": [4, 11, 14],
}
SUSPECT_TEST = ["8D5U5573.png", "8D5U5575.png"]   # 평가용에서만 빼고 한 번 더 보고 (D9)

# ── 채점 ──────────────────────────────────────────────────────
def judge(names, patches):
    """이름 (n, 24) + 24칸 색 (n, 24, 3) → 정답 · 일관성 (n, 24), 값은 1 · 0 · NaN

    채점할 칸은 "24칸 색이 있는 칸"으로 정한다 (빈칸 · 포화 의심 칸 = NaN, D3).
    조명 추정이 실패해 이름이 없으면(None) 빼지 않고 틀린 것으로 센다 (실패한 방법이 유리해지지 않게)
    """
    has = ~np.isnan(patches).any(axis=2)
    correct = np.full(names.shape, np.nan)
    consist = np.full(names.shape, np.nan)
    for i, k in zip(*np.nonzero(has)):
        correct[i, k] = names[i, k] in ALLOWED[k]
        consist[i, k] = names[i, k] == REF_NAMES[k]
    return correct, consist


def per_photo(mat, patches=None):
    """(n, 24) 1/0/NaN → 사진별 (맞은 수, 문제 수). patches: 칸 번호 목록(1부터), None이면 24칸 전부"""
    cols = [k - 1 for k in (patches or range(1, 25))]
    m = mat[:, cols]
    return np.nansum(m, axis=1), np.sum(~np.isnan(m), axis=1)


def rate(hits, total, sel=None):
    """사진별 (맞은 수, 문제 수) → 정답률 %. 문제를 모두 합쳐서 나눈다. sel: 사진 고르기 (bool)"""
    if sel is not None:
        hits, total = hits[sel], total[sel]
    return 100 * hits.sum() / total.sum() if total.sum() > 0 else np.nan


# ── 오차 범위: 사진 단위 부트스트랩 (D10 결정 3) ──────────────────
N_BOOT, SEED = 2000, 42


def _weights(cams, n_boot=N_BOOT, seed=SEED):
    """카메라별로 따로, 같은 장수를 중복 허용해 다시 뽑기 → 사진별 뽑힌 횟수 (n_boot, n)"""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, len(cams)))
    for cam in sorted(set(cams)):
        idx = np.nonzero(cams == cam)[0]
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return W


def bootstrap_rate(hits, total, cams):
    """정답률의 95% 구간 (%)"""
    W = _weights(cams)
    r = 100 * (W @ hits) / (W @ total)
    return tuple(np.percentile(r, [2.5, 97.5]))


def paired_diff(hits_a, hits_b, total, cams):
    """짝지은 비교: 같은 재표본에서 (A − B) 정답률 차이 → (차이 %p, 95% 하한, 상한)

    A와 B는 같은 사진 · 같은 칸을 채점하므로 문제 수(total)가 같다
    """
    W = _weights(cams)
    t = W @ total
    d = 100 * (W @ hits_a - W @ hits_b) / t
    diff = 100 * (hits_a.sum() - hits_b.sum()) / total.sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return diff, lo, hi


def rate_ci(hits, total, cams, sel=None):
    """정답률과 95% 구간 → (정답률 %, 하한, 상한, 사진 수). sel: 사진 고르기 (그 안에서 카메라별로 뽑기)"""
    if sel is not None:
        hits, total, cams = hits[sel], total[sel], cams[sel]
    if total.sum() == 0:
        return np.nan, np.nan, np.nan, 0
    lo, hi = bootstrap_rate(hits, total, cams)
    return 100 * hits.sum() / total.sum(), lo, hi, len(hits)


def boot_median(err, n_boot=N_BOOT, seed=SEED):
    """각도 오차 중앙값의 95% 구간 (사진 단위, 카메라 1종일 때)"""
    err = np.asarray(err, float)
    idx = np.random.default_rng(seed).integers(0, len(err), (n_boot, len(err)))
    return tuple(np.percentile(np.median(err[idx], axis=1), [2.5, 97.5]))


def paired_median_diff(a, b, n_boot=N_BOOT, seed=SEED):
    """같은 재표본에서 (A 중앙값 − B 중앙값) → (차이, 95% 하한, 상한). 음수면 A가 더 정확"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))
    d = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.median(a) - np.median(b)), lo, hi


# ── 최종 평가 잠금 (D5: 평가용은 한 번만) ──────────────────────────
def final_lock(out_md):
    """최종 1회 평가 전에 확인 → 지금 커밋 해시를 돌려준다. 조건이 안 맞으면 멈춘다

    ① 이미 실행했으면(결과 파일이 있으면) 멈춤 — 다시 돌려 좋은 숫자를 고르는 일을 막는다
    ② src · reports · splits에 커밋 안 된 변경이 있으면 멈춤 — 어떤 코드 · 설정으로 평가했는지 커밋 해시로 남기기 위해
       (reports/final은 최종 결과가 쌓이는 곳이라 빼고 본다)
    """
    if out_md.exists():
        raise SystemExit(f"⛔ 이미 최종 평가를 실행함: {out_md.relative_to(ROOT)}\n"
                         "   다시 돌려야 한다면 이유를 decisions.md에 먼저 적고 의논할 것 (결과 파일은 지우지 말 것)")
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    st = git("status", "--porcelain", "--", "src", "reports", "splits", ":(exclude)reports/final")
    if st.returncode != 0:
        raise SystemExit("⛔ git 상태를 확인할 수 없음 (레포 폴더에서 실행했는지 확인)")
    if st.stdout.strip():
        raise SystemExit("⛔ 커밋 안 된 변경이 있음 → 먼저 커밋 · 푸시하고 실행\n" + st.stdout)
    return git("rev-parse", "--short", "HEAD").stdout.strip()


def stamp(mode, commit):
    """결과 파일 머리에 붙일 한 줄: 언제 · 어떤 커밋으로 · 리허설인지 최종인지"""
    return f"- 실행: {datetime.now():%Y-%m-%d %H:%M} · 커밋 `{commit}` · {mode}"


# ── 조명 그룹: 정답 조명 → 색온도 (D10 결정 2) ────────────────────
# Adobe DNG Converter의 카메라 행렬 (XYZ → 카메라 원래 RGB, ×10000). LibRaw src/tables/colordata.cpp에서 옮김
ADOBE = {
    "EOS 5D":  [6347, -479, -972, -8297, 15954, 2480, -1968, 2131, 7649],
    "EOS-1D":  [6806, -179, -1020, -8097, 16415, 1687, -3267, 4236, 7690],
    "EOS-1Ds": [3925, 4060, -1739, -8973, 16552, 2545, -3287, 3945, 8243],
}
GROUPS = [("warm", "따뜻한 조명 <4000K", 0, 4000),
          ("daylight", "주광 4000~6500K", 4000, 6500),
          ("cool", "푸른 조명 >6500K", 6500, np.inf)]
GROUP_LABEL = {k: label for k, label, _, _ in GROUPS}


def illum_xy(illum, cams, how):
    """정답 조명 (n, 3) → 색좌표 xy (n, 2)

    how = "ours"  : 우리 행렬(color_matrix.json)에 그대로 넣기 (D10 원안)
    how = {"1D": "EOS-1Ds", "5D": "EOS 5D"} 처럼 카메라 → Adobe 행렬 이름 : 역행렬로 XYZ
    """
    illum = np.asarray(illum, float)
    xyz = np.full_like(illum, np.nan)
    if how == "ours":
        mats = load_color_matrices()
    else:
        mats = {cam: np.linalg.inv(np.array(ADOBE[model]).reshape(3, 3) / 10000) for cam, model in how.items()}
    for cam, M in mats.items():
        sel = cams == cam
        xyz[sel] = illum[sel] @ M.T
    return colour.XYZ_to_xy(xyz)


def cct_duv(xy):
    """xy (n, 2) → 색온도 K (McCamy, D10) · 흑체 궤적에서 떨어진 정도 Duv (Ohno 2013, 점검용)

    McCamy 식은 흑체 궤적 근처의 색에서만 의미가 있다 → |Duv|로 그 전제를 확인한다
    실제 조명(햇빛 · 백열등 · 형광등 · LED)은 대부분 |Duv| < 0.02
    """
    cct = colour.temperature.xy_to_CCT(xy, method="McCamy 1992")
    uv = colour.xy_to_UCS_uv(xy)
    duv = np.array([colour.temperature.uv_to_CCT(p, method="Ohno 2013")[1] for p in np.atleast_2d(uv)])
    return np.asarray(cct, float), duv


def to_group(cct):
    """색온도 (n,) → 그룹 키 (n,) — 경계값은 사진을 보기 전에 고정 (D10)"""
    out = np.empty(len(cct), dtype=object)
    for key, _, lo, hi in GROUPS:
        out[(cct >= lo) & (cct < hi)] = key
    return out


def illum_groups(illum, cams):
    """정답 조명 → 그룹 키. 계산 방법은 17이 reports/illum_params.json "cct"에 적은 것을 쓴다"""
    params = json.loads((ROOT / "reports" / "illum_params.json").read_text())
    how = params["cct"]["how"]
    cct, _ = cct_duv(illum_xy(illum, cams, how))
    return to_group(cct), cct


# ── 직접 실행하면: 부품 자기 점검 (데이터 없이) ─────────────────────
if __name__ == "__main__":
    # 1) 기준 이름: 정답표 첫 정답과 같고 5번만 "파랑" (D10)
    diff = [k + 1 for k in range(24) if REF_NAMES[k] != _ANS["answer"].iloc[k]]
    print(f"기준 이름이 정답표 첫 정답과 다른 칸: {diff} ({[REF_NAMES[k - 1] for k in diff]})")
    assert diff == [5] and REF_NAMES[4] == "파랑"
    assert all(REF_NAMES[k] in ALLOWED[k] for k in range(24)), "표준 색 이름이 정답 묶음 밖에 있음"
    # 2) 색온도: 흑체 색을 Adobe 행렬로 카메라 RGB로 바꿨다가 되돌리면 같은 색온도 · Duv ≈ 0
    T = np.array([2856.0, 4000, 5000, 6500, 9000])
    xy_true = np.array([colour.temperature.CCT_to_xy(t, method="Kang 2002") for t in T])
    xyz = colour.xy_to_XYZ(xy_true)
    A = np.array(ADOBE["EOS 5D"]).reshape(3, 3) / 10000
    raw = xyz @ A.T                                          # 그 조명 아래 흰 물체의 카메라 RGB
    cct, duv = cct_duv(illum_xy(raw, np.array(["5D"] * len(T)), {"5D": "EOS 5D"}))
    print("흑체 색온도 → 카메라 RGB → 되돌린 색온도:", dict(zip(T.astype(int), cct.round())), "· |Duv| 최대", f"{np.abs(duv).max():.4f}")
    assert np.allclose(cct, T, rtol=0.03) and np.abs(duv).max() < 0.003
    print("그룹:", list(to_group(cct)))
    # 3) 짝지은 부트스트랩: 같은 결과를 비교하면 차이 0, 구간도 0
    rng = np.random.default_rng(0)
    tot = rng.integers(20, 25, 50).astype(float)
    h = np.floor(tot * 0.7)
    cams = np.array(["1D"] * 10 + ["5D"] * 40)
    assert paired_diff(h, h, tot, cams) == (0.0, 0.0, 0.0)
    print("정답률 70% 근처 가짜 50장 → 95% 구간", np.round(bootstrap_rate(h, tot, cams), 1))
    # 4) 중앙값 부트스트랩: 같은 값끼리 비교하면 차이 0
    e = rng.gamma(2, 1.5, 200)
    assert paired_median_diff(e, e) == (0.0, 0.0, 0.0)
    print("가짜 각도 오차 200장 중앙값", round(float(np.median(e)), 2), "→ 95% 구간", np.round(boot_median(e), 2))
    print("✅ scoring.py 부품 점검 통과")