# app/check_web.py
# 목적: 웹 ② 내 사진 · ③ 실시간 카메라 부품의 자기 점검 (점검 10/30 C10)
#   - python app/engine.py 는 ① 데이터셋 모드(평가와 같은 계산, 600/600)를 점검하고 data/가 있어야 돈다
#   - 이 파일은 가짜 카메라 그림(fake_ipwebcam)만 써서 data/ 없이 어느 PC에서나 돈다
#   - engine.py의 ② · ③ 부품(조명 추정 · 보정 · 흰 종이 기준 · 안전장치 · 추정 불가 처리 · 보정 후 그림)을 고칠 때마다 돌린다
# 실행 (레포 폴더에서):  python app/check_web.py     → 마지막 줄이 ✅면 통과, ❌가 하나라도 있으면 실패(종료 코드 1)
#
# 점검(check)과 알려진 한계(known)를 나눈다
#   - check: 지금 맞게 동작해야 하는 것. 고친 뒤 ❌가 나오면 무언가 깨진 것
#   - known: 점검에서 찾았지만 아직 안 고친 한계(C1 · C2 · C3). 지금은 "한계 그대로"가 정상이고,
#            고쳐서 한계가 사라지면 🔔로 알려 준다 → 그때 이 줄을 check로 바꾼다
# 숫자는 시험용 가짜 그림의 숫자이고 평가 결과가 아니다 (평가 숫자는 reports/final/)
import csv
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import colour

warnings.filterwarnings("ignore")                       # colour의 matplotlib 안내 문구 숨김
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import engine as E                             # noqa: E402  (평가 부품을 그대로 쓰는 웹 계산)
from app import fake_ipwebcam as F                      # noqa: E402  (정답을 아는 가짜 카메라 그림)

# 24칸 정답표 (애매한 칸은 두 이름 모두 정답, D2)
ANSWER = {int(r["patch"]): {r["answer"], r["answer_alt"]} - {""}
          for r in csv.DictReader(open(ROOT / "reports" / "answer_table.csv", encoding="utf-8"))}
LIGHT = dict(F.LIGHTS)                                  # {"주광": 이득, "따뜻한 조명": …, "푸른 조명": …}
PAPER = (slice(415, 475), slice(290, 350))              # 장면 아래 빈 곳에 놓는 종이 자리 (60×60, 칸과 안 겹침)
PAPER_XY = (320 / F.W, 445 / F.H)                       # 종이 가운데 (비율)
BG_XY = (20 / F.W, 20 / F.H)                            # 회색 배경 (18%)

n_ok = n_bad = n_known = 0
results = []                                            # 모든 결과 → 마지막에 JSON으로 보낼 수 있는지 확인


# ── 도우미 ────────────────────────────────────────────────────
def check(title, ok, detail=""):
    global n_ok, n_bad
    n_ok, n_bad = n_ok + bool(ok), n_bad + (not ok)
    print(f"  {'✅' if ok else '❌'} {title}" + (f" — {detail}" if detail else ""))


def known(ref, title, still, detail):
    """still = 한계가 아직 그대로인가. 그대로면 ⚠(정상), 사라졌으면 🔔(고쳤다면 check로 바꾸기)"""
    global n_known
    n_known += 1
    if still:
        print(f"  ⚠ [{ref}] {title} — 한계 그대로: {detail}")
    else:
        print(f"  🔔 [{ref}] {title} — 한계가 더 이상 안 보여요: {detail} → 고쳤다면 이 줄을 check로 바꾸세요")


def keep(res):
    results.append(res)
    return res


def frame(gain, scene=None, seed=0):
    """가짜 카메라 한 장 → 웹이 받는 것과 같은 8비트 sRGB 배열 (잡음 seed를 고정해 매번 같은 그림)"""
    return E.load_frame(F.render_jpeg(gain, scene, np.random.default_rng(seed)))


def live(fr, k_or_xy, **kw):
    xy = F.patch_center(k_or_xy) if isinstance(k_or_xy, int) else k_or_xy
    return keep(E.analyze_live(fr, *xy, **kw))


def white_at(fr, k_or_xy):
    xy = F.patch_center(k_or_xy) if isinstance(k_or_xy, int) else k_or_xy
    return keep(E.measure_white(fr, *xy))


def paper_scene(rgb8):
    """회색 배경 + 컬러체커 장면 아래에 종이 한 장 (rgb8 = 주광에서 보이는 종이 색, 8비트 sRGB)"""
    s = F.SCENE.copy()
    s[PAPER] = colour.cctf_decoding(np.array(rgb8) / 255, function="sRGB")
    return s


def lab_of(srgb8, xy, r=E.ROI_R):
    """화면 그림의 한 점 둘레 → Lab (이름 계산과 같은 길: 선형 평균 → sRGB 행렬 + D65→D50 → Lab)"""
    h, w = srgb8.shape[:2]
    x1, y1, x2, y2 = E.roi_box(xy[0] * w, xy[1] * h, w, h, r)
    lin = colour.cctf_decoding(srgb8[y1:y2, x1:x2].reshape(-1, 3) / 255, function="sRGB").mean(axis=0)
    return E.to_lab(E.SRGB_TO_XYZ50 @ lin)


# ── 1. 보정 흐름: 세 조명에서 24칸 ──────────────────────────────────
print("1. 보정 흐름 — 가짜 카메라 24칸, 화이트 밸런스를 고정한 것처럼 조명 색이 그대로 찍힘")
count = {}
for name, gain in F.LIGHTS:
    fr = frame(gain)
    il = E.upload_illum(fr)
    w = white_at(fr, 19)                                 # 컬러체커 흰 칸(19번)으로 흰 종이 기준 맞추기
    before = sog = white = 0
    wrong = []
    for k in range(1, 25):
        r = live(fr, k, illum=il)
        before += r["baseline_name"] in ANSWER[k]
        sog += r["color_name"] in ANSWER[k]
        if r["color_name"] not in ANSWER[k]:
            wrong.append(f"{k}번 → {r['color_name']}")
        if w["ok"]:
            white += live(fr, k, white=w, illum=il)["color_name"] in ANSWER[k]
    count[name] = (before, sog, white)
    print(f"  · {name}: 보정 전 {before}/24 · Shades of Gray {sog}/24 · 흰 칸 기준 {white}/24"
          + (f"  (SoG 틀린 칸: {', '.join(wrong)})" if wrong else ""))
    check(f"{name}: 흰 칸으로 맞추기가 안전장치를 통과", w["ok"], f"자동 추정과 {w.get('checks', {}).get('angle')}° 차이")
    check(f"{name}: Shades of Gray 23/24 이상", sog >= 23)
    check(f"{name}: 흰 칸 기준 24/24", white == 24)
b, s, _ = count["따뜻한 조명"]
check("따뜻한 조명에서 보정 효과가 보임 (보정 전 < Shades of Gray)", b < s, f"{b} → {s}")
print("  · 19번 흰색이 SoG로 '회색'인 건 폰 밝기를 그대로 쓰는 한계(웹-2 ②) — 흰 종이 기준이 푸는 문제")

# ── 2. 보정 후 그림 = 이름 계산 ─────────────────────────────────────
print("2. 보정 후 그림 — 그림에서 다시 잰 이름이 오른쪽 이름과 같아야 함 (correct_image)")
fr = frame(LIGHT["따뜻한 조명"])
il = E.upload_illum(fr)
w = white_at(fr, 19)
for label, kw in [("Shades of Gray", {}), ("흰 종이 기준", {"white": w})]:
    shown = E.correct_image(fr, il, kw.get("white"))
    same = sum(E.describe(lab_of(shown, F.patch_center(k)))["name"] == live(fr, k, illum=il, **kw)["color_name"]
               for k in range(1, 25))
    check(f"{label}: 보정 후 그림의 24칸 이름 = 계산한 이름", same == 24, f"{same}/24")
c_before = float(E.lab_to_lch(lab_of(fr, BG_XY))[1])
c_after = float(E.lab_to_lch(lab_of(E.correct_image(fr, il), BG_XY))[1])
check("따뜻한 조명의 회색 배경이 보정 후 회색 쪽으로 (채도 줄어듦)", c_after < c_before / 3, f"채도 {c_before:.1f} → {c_after:.1f}")

# ── 3. 흰 종이 안전장치 ①~⑥ ────────────────────────────────────────
print("3. 흰 종이 안전장치 — 일부러 틀리게 맞춰서 걸리는지 (①~③ 거절 · ④⑤ 확인 필요 · ⑥ 경고)")
hot = F.SCENE.copy()
y0, x0 = int(F.patch_center(19)[1] * F.H), int(F.patch_center(19)[0] * F.W)
hot[y0 - 40:y0 + 40, x0 - 40:x0 + 40] *= 1.6                       # 흰 칸만 1.6배 → 하얗게 날아감
r = white_at(frame(LIGHT["주광"], hot), 19)
check("① 하얗게 날아간 종이 → 거절", not r["ok"] and r["reason"] == "too_bright" and not r.get("confirm"), r["reason"])
r = white_at(fr, 24)
check("② 검정 칸 → 거절 (너무 어두움)", not r["ok"] and r["reason"] == "too_dark" and not r.get("confirm"), r["reason"])
edge = ((F.X0 + F.CW + F.GAP / 2) / F.W, F.patch_center(19)[1])   # 19번과 20번 사이 → 네모 안에 흰색 · 회색 · 배경이 섞임
r = white_at(fr, edge)
check("③ 칸 경계 → 거절 (고르지 않음)", not r["ok"] and r["reason"] == "not_uniform" and not r.get("confirm"), r["message"][:30])
r = white_at(fr, 15)
check("④ 빨강 칸 → 확인 필요 (흰색이 아닐 수 있음)",
      not r["ok"] and r.get("confirm") and r["reason"] == "not_white" and "ref" in r, r["message"][:28])
r = white_at(fr, BG_XY)
check("⑤ 회색 배경 → 확인 필요 (회색 물체일 수 있음)",
      not r["ok"] and r.get("confirm") and r["reason"] == "not_white_gray" and "ref" in r, r["message"][:28])
fr_blue = frame(LIGHT["푸른 조명"])
r = live(fr_blue, 21, white=w, illum=E.upload_illum(fr_blue))
check("⑥ 따뜻한 조명에서 맞춘 기준을 푸른 조명에서 쓰면 → 경고", r["white_warning"] is not None,
      f"21번 회색 칸이 '{r['color_name']}' + 경고")
r = live(fr, 21, white=w, illum=il)
check("⑥ 같은 조명에서는 경고 없음", r["white_warning"] is None)

# ── 4. 조명을 추정할 수 없는 장 (NaN) ──────────────────────────────
print("4. 추정 불가 장 — 새까맣거나 전부 하얗게 날아간 장에서 서버 오류 없이 '판단하기 어려워요'")
for label, val in [("새까만 장", 0), ("전부 포화된 장", 255)]:
    bad = np.full((F.H, F.W, 3), val, np.uint8)
    est = E.upload_illum(bad)
    r1 = live(bad, (0.5, 0.5))                                     # 서버 첫 장처럼 illum=None → 안에서 추정
    r2 = live(bad, (0.5, 0.5), illum=est)
    ok = all(r["status"] == "uncertain" and r["reason"] == "no_illum" for r in (r1, r2))
    check(f"{label} → 판단 어려움 (조명 추정 불가)", ok and not np.all(np.isfinite(est)), r1["message"])
    check(f"{label}: 보정 후 그림은 원본 그대로", E.correct_image(bad, est) is bad)
v = np.ones(3) / np.sqrt(3)
nan3 = np.full(3, np.nan)
check("부드럽게: 첫 장부터 추정 불가 → 아직 없음(None)", E.smooth_illum(None, nan3) is None)
check("부드럽게: 추정 불가 장 → 이전 값 유지", np.allclose(E.smooth_illum(v, nan3), v))
near = np.array([1.0, 1.0, 1.1]); near /= np.linalg.norm(near)    # v에서 약 3°
mix = E.smooth_illum(v, near)
check("부드럽게: 조금 다르면 30%만 따라감", 0 < E.angle_deg(v, mix) < E.angle_deg(v, near) * 0.5,
      f"{E.angle_deg(v, near):.1f}° 중 {E.angle_deg(v, mix):.1f}° 이동")
far = np.array([1.0, 0.8, 0.5]); far /= np.linalg.norm(far)      # v에서 10° 넘게
check("부드럽게: 10° 넘게 다르면 바로 바꿈", np.allclose(E.smooth_illum(v, far), far),
      f"{E.angle_deg(v, far):.1f}° 차이")

# ── 5. 알려진 한계 (10/30 점검 C1 · C2 · C3, 아직 안 고침) ─────────────
print("5. 알려진 한계 — 지금은 '한계 그대로'가 정상. 고친 뒤 🔔가 나오면 check로 바꾼다")
day = LIGHT["주광"]
fr_w = frame(day, paper_scene((243, 243, 243)))
w_ref = white_at(fr_w, PAPER_XY)
for label, rgb8 in [("크림 종이", (245, 232, 200)), ("베이지 종이", (230, 215, 185))]:
    frc = frame(day, paper_scene(rgb8))
    wc = white_at(frc, PAPER_XY)
    g21 = live(frc, 21, white=wc, illum=E.upload_illum(frc))["color_name"] if wc["ok"] or wc.get("ref") else "-"
    known("C1", f"{label}가 ④ 흰색 확인을 통과", wc["ok"],
          f"자동 추정과 {E.angle_deg(wc['ref'], wc['sog']):.1f}° (흰 종이 {w_ref['checks']['angle']}°) → 기준으로 쓰면 21번 회색 칸이 '{g21}'")
yellow = np.empty_like(F.SCENE)
yellow[:] = colour.cctf_decoding(np.array([235, 200, 60]) / 255, function="sRGB")   # 노란 포스트잇이 화면 가득
fry = frame(day, yellow)
r = live(fry, (0.5, 0.5))
wy = white_at(fry, (0.5, 0.5))
known("C2", "노란 면이 화면을 채우면 SoG와 흰 종이 확인이 함께 속음",
      r["color_name"] != "노랑" and wy["ok"], f"SoG 이름 '{r['color_name']}' · 흰 종이로 맞추기 {'통과' if wy['ok'] else '걸림'}")
fr_bright = frame(day, F.SCENE * 1.8)                             # 물건을 볼 때 폰이 자동 노출로 밝게 올린 장
r = live(fr_bright, 24, white=white_at(frame(day), 19), illum=E.upload_illum(fr_bright))
known("C3", "흰 종이를 맞춘 뒤 자동 노출로 밝아지면 검정이 검정이 아님 · 경고 없음",
      r["color_name"] != "검정" and r["white_warning"] is None,
      f"24번 검정 칸이 '{r['color_name']}'(밝기 {r['lab']['L']}) · ⑥ 경고 {'없음' if r['white_warning'] is None else '있음'}")

# ── 6. 서버가 답을 보낼 수 있는지 ──────────────────────────────────
print("6. JSON — 위에서 나온 결과를 모두 서버 응답(JSON)으로 보낼 수 있어야 함 (NaN · 무한대 없음)")
bad = []
for i, res in enumerate(results):
    try:
        json.dumps(res, allow_nan=False)
    except (TypeError, ValueError) as e:
        bad.append(f"{i}: {e}")
check(f"결과 {len(results)}개 모두 JSON으로 보낼 수 있음", not bad, "; ".join(bad[:3]))

print()
if n_bad:
    print(f"❌ check_web.py 점검 실패 — 통과 {n_ok} · 실패 {n_bad} (알려진 한계 {n_known}개는 따로)")
    sys.exit(1)
print(f"✅ check_web.py 점검 통과 — {n_ok}개 (알려진 한계 {n_known}개는 위에 ⚠로 표시)")