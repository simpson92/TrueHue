# src/21_final_gehler.py
# 목적: Gehler-Shi 평가용 378장으로 색 이름 채점을 최종 1회 한다 — 프로젝트의 주 결과 (D5 · D6 · D9 · D10)
#   한 번에 하는 일: ① 사진마다 조명 추정(보정 3종, 18과 같은 계산) ② 기준선 · 보정 3종 · 상한선 채점(19와 같은 계산)
#                    ③ D10 보고 범위 전부 + 95% 구간 ④ 의심 2장 제외 376장 결과도 함께 (D9)
#   - 주 결과(미리 정함, D10 결정 4): 보정 3종 각각 − 기준선, 전체 24칸 정답률 차이 (95% 구간)
#   - 설정(정답표 · 이름 규칙 · 행렬 · 기준선 조명 · p · 색온도 행렬)은 모두 보정용에서 정해 고정. 평가용으로 고르는 것 없음
#   - 오차 범위: 사진 단위 부트스트랩 2,000회 · 95% · seed 42 · 카메라별로 따로 뽑기, 방법 비교는 같은 재표본 (D10 결정 3)
#   실행 (둘 중 하나를 꼭 적어야 돈다)
#     python src/21_final_gehler.py --rehearsal   ← 보정용 190장으로 리허설. 19와 같은 정답률이 나와야 정상 (평가용 안 엶)
#                                                   의심 사진 제외 경로는 보정용의 8D5U5541로 대신 시험
#     python src/21_final_gehler.py --final       ← 평가용 최종 1회. 커밋 안 된 변경이 있거나 이미 실행했으면 멈춤
#   결과: --final     → reports/final/gehler_test.md (깃에 올리는 최종 기록) + outputs/final/gehler_test_*.csv
#         --rehearsal → outputs/final_rehearsal/gehler_calib.md · *.csv
import json
import sys
import time
import numpy as np
import pandas as pd
from common import ROOT, NEUTRALS, angular_error, load_gehler_index, load_gehler_image
from illuminant import PARAMS, load_methods, sog_many, summarize
from pipeline import load_color_matrices, patches_to_lab, name_patches
from scoring import (judge, per_photo, rate, rate_ci, paired_diff, illum_groups, final_lock, stamp,
                     SUBSETS, GROUPS, REF_NAMES, SUSPECT_TEST)

if "--final" in sys.argv:
    mode, split, suspects = "최종 1회 (평가용)", "test", SUSPECT_TEST
    out_md = ROOT / "reports" / "final" / "gehler_test.md"
    out_dir = ROOT / "outputs" / "final"
    commit = final_lock(out_md)                                  # 이미 실행 / 커밋 안 된 변경 → 여기서 멈춤
    print("⚠️ Gehler-Shi 평가용 378장을 엽니다 — 최종 1회 (D5)")
elif "--rehearsal" in sys.argv:
    mode, split, suspects = "리허설 (보정용, 최종 숫자 아님)", "calib", ["8D5U5541.png"]
    out_md = ROOT / "outputs" / "final_rehearsal" / "gehler_calib.md"
    out_dir = out_md.parent
    commit = "리허설"
else:
    raise SystemExit("사용법: python src/21_final_gehler.py --rehearsal  (보정용 리허설)  |  --final  (평가용 최종 1회)")
out_md.parent.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

# ── 1. 데이터 · 고정된 설정 ───────────────────────────────────────
d = load_gehler_index(split)
names_img, cams, P = d["names"], d["cams"], d["patches"]
n = len(names_img)
params = json.loads(PARAMS.read_text())
methods = load_methods("gehler")                       # {"gray_world": 1, "white_patch": inf, "shades_of_gray": p}
p_sog = methods["shades_of_gray"]
print(f"Gehler-Shi {split} {n}장 (1D {np.sum(cams == '1D')} · 5D {np.sum(cams == '5D')}), p = {p_sog}, "
      f"색온도 행렬 {params['cct']['how']}")

# ── 2. 조명 추정 (18과 같은 계산: 가림 영역 · 포화 제외 → Shades of Gray 일반식) ──────
ps = sorted(set(methods.values()))
est = {m: np.full((n, 3), np.nan) for m in methods}
valid_ratio = np.zeros(n)
t0 = time.time()
for k in range(n):
    lin, valid = load_gehler_image(names_img[k])
    e = sog_many(lin, valid, ps)
    for m, p in methods.items():
        est[m][k] = e[p]
    valid_ratio[k] = valid.mean()
    if (k + 1) % 40 == 0 or k + 1 == n:
        print(f"  조명 추정 {k + 1:3d}/{n} ({time.time() - t0:.0f}초)")

BASE, UPPER = "기준선(고정)", "상한선(정답)"
ILLUM = {
    BASE: np.array([params["baseline"][c] for c in cams]),
    "Gray World": est["gray_world"],
    "White Patch": est["white_patch"],
    f"SoG(p={p_sog})": est["shades_of_gray"],
    UPPER: d["illum"],
}
CORR = [m for m in ILLUM if m not in (BASE, UPPER)]

# ── 3. 채점 (19와 같은 계산) ──────────────────────────────────────
groups, cct = illum_groups(d["illum"], cams)
no_gray = int(np.isnan(P[:, [k - 1 for k in NEUTRALS], :]).any(axis=2).all(axis=1).sum())
n_fail = int(sum(np.isnan(est[m]).any(axis=1).sum() for m in methods))
mats = load_color_matrices()
R = {}
for m, il in ILLUM.items():
    lab = patches_to_lab(P, il, cams, mats)
    nm = name_patches(lab)
    correct, consist = judge(nm, P)
    R[m] = {"names": nm, "lab": lab, "correct": correct, "consist": consist}
keep = ~np.isin(names_img, suspects)                     # 의심 사진 제외 (D9)
n_q = int((~np.isnan(P).any(axis=2)).sum())

# 보고 범위 (D10 결정 1): 칸 묶음 · 카메라 · 조명 그룹
COLS = [(label, ks, None) for label, ks in SUBSETS.items()]
COLS += [(f"{cam} 카메라", None, cams == cam) for cam in ["1D", "5D"]]
COLS += [(label, None, groups == key) for key, label, _, _ in GROUPS]

# ── 4. 보고 ───────────────────────────────────────────────────
L = [f"# Gehler-Shi 색 이름 채점 — {mode}", "",
     stamp(mode, commit),
     f"- 사진 {n}장 (1D {np.sum(cams == '1D')} · 5D {np.sum(cams == '5D')}) · 채점 문제 {n_q}칸 · "
     f"의심 사진 제외 시 {keep.sum()}장 (제외: {', '.join(s for s in suspects if s in names_img) or '없음'})",
     f"- 회색 칸 20~23번이 모두 빈 사진 {no_gray}장 (B3) · 조명 추정 실패(쓸 픽셀 없음) {n_fail}건 (틀린 것으로 셈)",
     "- 조명 그룹(정답 조명 색온도): " + " · ".join(f"{label} {np.sum(groups == key)}장" for key, label, _, _ in GROUPS),
     f"- 설정: Shades of Gray p = {p_sog} · 기준선 = 보정용 카메라별 평균 조명 · 색온도 행렬 {params['cct']['how']} "
     "(모두 보정용에서 고정, `reports/illum_params.json`)",
     "- 오차 범위: 사진 단위 부트스트랩 2,000회 · 95% 백분위 · seed 42 · 카메라별로 따로 뽑기. 방법 비교는 같은 재표본에서의 차이", ""]


def primary(sel, title):
    """주 결과: 보정 3종(+ 상한선 참고) − 기준선, 전체 24칸"""
    out = [f"### {title}", "", "| 비교 | 정답률 차이 [95% 구간] | 판단 |", "|---|---|---|"]
    hb, tot = per_photo(R[BASE]["correct"])
    for m in CORR + [UPPER]:
        hm, _ = per_photo(R[m]["correct"])
        dd, lo, hi = paired_diff(hm[sel], hb[sel], tot[sel], cams[sel])
        verdict = "0 포함 안 함 → 의미 있는 차이" if lo > 0 or hi < 0 else "0 포함 → 차이 있다고 말할 수 없음"
        tag = " (참고)" if m == UPPER else ""
        out.append(f"| {m} − 기준선{tag} | **{dd:+.1f}%p** [{lo:+.1f}, {hi:+.1f}] | {verdict} |")
    return out + [""]


L += ["## 주 결과 (미리 정함, D10 결정 4) — 보정 3종 − 기준선, 전체 24칸 정답률", ""]
L += primary(np.ones(n, bool), f"전체 {n}장")
L += primary(keep, f"의심 사진 제외 {keep.sum()}장 (D9)")


def metric_table(metric, sel_all, title):
    out = [f"### {title}", "", "| 방법 | " + " | ".join(c[0] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for m in ILLUM:
        cells = []
        for _, ks, sel in COLS:
            h, t = per_photo(R[m][metric], ks)
            s = sel_all if sel is None else (sel & sel_all)
            r, lo, hi, _ = rate_ci(h, t, cams, s)
            cells.append("-" if np.isnan(r) else f"{r:.1f} [{lo:.1f}, {hi:.1f}]")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    cnt = []
    for _, ks, sel in COLS:
        _, t = per_photo(R[BASE]["correct"], ks)
        s = sel_all if sel is None else (sel & sel_all)
        cnt.append(f"{int(t[s].sum())}칸 · {int(s.sum())}장")
    out.append("| 문제 수 | " + " | ".join(cnt) + " |")
    return out + [""]


ALL = np.ones(n, bool)
L += ["## 정답률 % [95% 구간] — 정답 묶음 {answer, answer_alt} 안이면 정답", ""]
L += metric_table("correct", ALL, f"전체 {n}장")
L += metric_table("correct", keep, f"의심 사진 제외 {keep.sum()}장")
L += ["## 이름 일관성 % [95% 구간] — 공식 표준 색에 규칙을 넣은 이름과 같은 비율 (보조)", ""]
L += metric_table("consist", ALL, f"전체 {n}장")

# 보조: 보고 범위마다 보정 3종 − 기준선 차이
L += ["## 보조 결과 — 보고 범위별 정답률 차이 (방법 − 기준선, %p [95% 구간])", "",
      "| 방법 | " + " | ".join(c[0] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
for m in CORR + [UPPER]:
    cells = []
    for _, ks, sel in COLS:
        s = ALL if sel is None else sel
        if s.sum() == 0:
            cells.append("-")
            continue
        hb, tot = per_photo(R[BASE]["correct"], ks)
        hm, _ = per_photo(R[m]["correct"], ks)
        dd, lo, hi = paired_diff(hm[s], hb[s], tot[s], cams[s])
        cells.append(f"{dd:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    L.append(f"| {m} | " + " | ".join(cells) + " |")
L.append("")

# 조명 추정 각도 오차 (평가용)
L += ["## 조명 추정 각도 오차 (도) — 평가용 사진, 가림 영역 · 포화 제외", "",
      "| 방법 | 중앙값 | 1D 중앙값 | 5D 중앙값 | 평균 | 나쁜 25% |", "|---|---|---|---|---|---|"]
for m in [BASE] + CORR:
    err = angular_error(ILLUM[m], d["illum"])
    s = summarize(err)
    L.append(f"| {m} | {s['median']:.2f} | {np.nanmedian(err[cams == '1D']):.2f} | {np.nanmedian(err[cams == '5D']):.2f} | "
             f"{s['mean']:.2f} | {s['worst25']:.2f} |")
L += ["", f"쓸 수 있는 픽셀 비율: 중앙값 {np.median(valid_ratio):.0%} · 최저 {valid_ratio.min():.0%}", ""]

# 칸별 정답률 · 오답
L += ["## 칸별 정답률 %", "", "| 칸 | 기준 이름 | " + " | ".join(ILLUM) + " | 가장 많은 오답 (상한선 · SoG) |",
      "|---|---|" + "---|" * len(ILLUM) + "---|"]
sog = CORR[-1]
for k in range(24):
    cells = [f"{100 * np.nanmean(R[m]['correct'][:, k]):.0f}" for m in ILLUM]
    def top_wrong(m):
        c, nm = R[m]["correct"][:, k], R[m]["names"][:, k]
        w = pd.Series([x if x is not None else "(없음)" for x in nm[c == 0]])
        return f"{w.value_counts().index[0]} {w.value_counts().iloc[0]}" if len(w) else "-"
    L.append(f"| {k + 1} | {REF_NAMES[k]} | " + " | ".join(cells) + f" | {top_wrong(UPPER)} · {top_wrong(sog)} |")
L.append("")

# 회색 칸 채도 (D4 예측: 조명 1°당 약 2.4)
L += ["## 회색 칸 21 · 22번 채도 중앙값 (기준 10보다 크면 유채색으로 틀림)", "", "| 방법 | 채도 중앙값 | 10 넘는 비율 |", "|---|---|---|"]
for m in ILLUM:
    lab = R[m]["lab"][:, [20, 21], :]
    ch = np.hypot(lab[..., 1], lab[..., 2])
    L.append(f"| {m} | {np.nanmedian(ch):.1f} | {np.nanmean(ch > 10):.0%} |")
hu, tu = per_photo(R[UPPER]["correct"])
L += ["", f"카메라 변환 손실 (100% − 상한선, 전체 24칸): {100 - rate(hu, tu):.1f}%p (D6)", ""]

out_md.write_text("\n".join(L) + "\n", encoding="utf-8")

# ── 5. 저장 · 화면 요약 ─────────────────────────────────────────
rows = []
for m in ILLUM:
    nm, c, cs = R[m]["names"], R[m]["correct"], R[m]["consist"]
    for i, k in zip(*np.nonzero(~np.isnan(c))):
        rows.append({"image": names_img[i], "camera": cams[i], "group": groups[i], "cct": round(float(cct[i])),
                     "suspect": bool(not keep[i]), "patch": k + 1, "method": m, "name": nm[i, k],
                     "correct": int(c[i, k]), "consistent": int(cs[i, k])})
pd.DataFrame(rows).to_csv(out_dir / f"gehler_{split}_patches.csv", index=False)
pd.DataFrame({"image": names_img, "camera": cams, "group": groups, "cct": cct.round(), "valid_ratio": valid_ratio.round(3),
              **{f"{m}_{ch}": est[m][:, j] for m in methods for j, ch in enumerate("rgb")}}
             ).to_csv(out_dir / f"gehler_{split}_illum.csv", index=False)

print("\n[정답률 · 전체 24칸]")
hb, tot = per_photo(R[BASE]["correct"])
for m in ILLUM:
    h, _ = per_photo(R[m]["correct"])
    r, lo, hi, _ = rate_ci(h, tot, cams)
    print(f"  {m:14s} {r:5.1f}%  [{lo:5.1f}, {hi:5.1f}]")
print(f"\n[주 결과 · 전체 {n}장] 보정 3종 − 기준선, 전체 24칸")
print("\n".join(primary(ALL, "")[4:-1]))
print(f"\n[주 결과 · 의심 사진 제외 {keep.sum()}장]")
print("\n".join(primary(keep, "")[4:-1]))
if split == "calib":
    print("\n※ 리허설 확인: 전체 24칸 정답률이 19 출력과 같아야 정상 — 74.1 · 80.2 · 64.1 · 86.0 · 95.8, 차이 +6.1 · −10.1 · +11.8")
print(f"\n저장: {out_md.relative_to(ROOT)} (전체 보고), {(out_dir / f'gehler_{split}_patches.csv').relative_to(ROOT)}")