# src/11_detect_charts.py
# 목적: Gehler 568장 모두에서 컬러체커 위치를 자동으로 찾고(09의 3차 방식), 사람이 확인할 그림을 만든다 (D8)
#   ① 사진마다 검출 상자 → 넉넉한 "가림 영역"(상자 긴 변의 25%씩 여유, B1)
#   ② 30장씩 한눈에 보기 그림: 초록 = 검출 상자, 노랑 = 실제로 가릴 영역 → 노랑이 컬러체커를 다 덮는지 확인
#   ③ 검출 실패한 사진은 멈추지 않고 "fail"로 표시해 맨 앞 그림에 모음 (B2)
#   ④ 정답 조명이 의심스러운 3장(D9)은 따로 크게 잘라 저장 → 원인 확인용 (B4)
#   ⑤ 회색 칸(20~23번)이 모두 빈칸인 사진 수를 센다 (B3: 이런 사진은 노출 맞추기가 안 됨)
#   - 평가용 사진도 처리한다: 가림 영역은 평가 때 필요하고, 여기서는 위치만 찾을 뿐 채점 · 조명 추정은 하지 않음
#   - 오래 걸리므로 중간에 멈춰도 이어서 할 수 있다 (outputs/charts/auto_boxes.csv에 한 장씩 저장)
#   결과: outputs/charts/ (그림 · 자동 결과)  +  reports/chart_boxes.csv (검수 결과의 기준 파일, 처음 한 번만 만듦)
import time
import cv2
import numpy as np
import pandas as pd
from common import ROOT, GEHLER_DIR, GEHLER_BLACK, NEUTRALS, read_rgb, load_gehler_index
from chart import detect_chart, mask_box, to_display

OUT = ROOT / "outputs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)
AUTO = OUT / "auto_boxes.csv"                     # 자동 검출 결과 (이어서 하기용)
FINAL = ROOT / "reports" / "chart_boxes.csv"      # 사람이 검수 · 수정하는 기준 파일 (깃에 올림)
SUSPECT = ["8D5U5541.png", "8D5U5573.png", "8D5U5575.png"]   # D9
TW, TH = 480, 320                                  # 한눈에 보기 그림의 사진 한 장 크기

d = load_gehler_index()                            # 568장 전체 (24칸 색 · 정답 조명)
calib = set((ROOT / "splits" / "gehler_calib.txt").read_text().split())
n = len(d["names"])

# ── B3: 회색 칸 20~23번이 모두 빈칸인 사진 (데이터 값만 확인) ──────────
nidx = [k - 1 for k in NEUTRALS]
no_gray = [d["names"][i] for i in range(n) if np.isnan(d["patches"][i, nidx]).any(axis=1).all()]
print(f"[B3] 회색 칸 20~23번이 모두 빈칸인 사진: {len(no_gray)}장 {no_gray if no_gray else ''}")


def load(i):
    """i번째 사진 → 블랙 레벨 뺀 선형 RGB (float)"""
    raw = read_rgb(GEHLER_DIR / "png" / d["names"][i]).astype(np.float64)   # ⚠️ RGB 순서로 읽기
    return np.clip(raw - GEHLER_BLACK[d["cams"][i]], 0, None)


# ── ① 자동 검출 (이미 한 사진은 건너뜀) ─────────────────────────────
done = pd.read_csv(AUTO) if AUTO.exists() else pd.DataFrame()
done_names = set(done["image"]) if len(done) else set()
rows = done.to_dict("records") if len(done) else []
t0 = time.time()
for i in range(n):
    name = d["names"][i]
    if name in done_names:
        continue
    lin = load(i)
    box, found, win = detect_chart(lin, d["patches"][i])
    row = {"no": i + 1, "image": name, "camera": d["cams"][i],
           "split": "calib" if name in calib else "test",
           "status": "ok" if box is not None else "fail",
           "found": round(float(found), 2), "window": win}
    if box is not None:
        bx = np.round(box).astype(int)
        mx = mask_box(box, lin.shape)
        row.update(det_x1=bx[0], det_y1=bx[1], det_x2=bx[2], det_y2=bx[3],
                   x1=mx[0], y1=mx[1], x2=mx[2], y2=mx[3])
    rows.append(row)
    pd.DataFrame(rows).to_csv(AUTO, index=False)   # 한 장마다 저장 → 중간에 멈춰도 이어서 가능
    k = len(rows)
    if k % 20 == 0 or k == n:
        eta = (time.time() - t0) / max(k - len(done_names), 1) * (n - k)
        print(f"  {k:3d}/{n} 처리 (남은 시간 약 {eta / 60:.0f}분)")

auto = pd.DataFrame(rows).sort_values("no").reset_index(drop=True)
print(f"\n자동 검출: 성공 {(auto.status == 'ok').sum()}장 / 실패 {(auto.status == 'fail').sum()}장")

# ── 기준 파일 만들기 (이미 있으면 사람이 고친 내용을 지키려고 덮어쓰지 않음) ──
if FINAL.exists():
    print(f"{FINAL.relative_to(ROOT)} 이미 있음 → 덮어쓰지 않음 (사람이 고친 내용 보호)")
    final = pd.read_csv(FINAL)
else:
    final = auto[["no", "image", "camera", "split", "status", "x1", "y1", "x2", "y2"]].copy()
    final["source"] = np.where(final["status"] == "ok", "auto", "none")   # auto / manual / none
    final["note"] = ""
    final.to_csv(FINAL, index=False)
    print(f"저장: {FINAL.relative_to(ROOT)} (가림 영역 기준 파일, 다음 단계: 그림 확인 → 12로 수정)")


# ── ② 한눈에 보기 그림 ──────────────────────────────────────────────
def thumb(i, r):
    """사진 i의 작은 그림 + 검출 상자(초록) + 가림 영역(노랑) + 번호"""
    lin = load(i)
    H, W = lin.shape[:2]
    small = cv2.resize(lin.astype(np.float32), (TW, TH), interpolation=cv2.INTER_AREA)
    gains = d["illum"][i] / d["illum"][i][1]
    scale = 0.9 / np.percentile(small / gains, 99)
    vis = (to_display(small, d["illum"][i], scale)[:, :, ::-1] * 255).astype(np.uint8).copy()   # 저장용 BGR
    sx, sy = TW / W, TH / H
    if r["status"] == "ok" and not pd.isna(r.get("det_x1")):
        cv2.rectangle(vis, (int(r["det_x1"] * sx), int(r["det_y1"] * sy)),
                      (int(r["det_x2"] * sx), int(r["det_y2"] * sy)), (0, 255, 0), 1)
    f = final.loc[final["image"] == r["image"]].iloc[0]
    if not pd.isna(f["x1"]):
        cv2.rectangle(vis, (int(f["x1"] * sx), int(f["y1"] * sy)),
                      (int(f["x2"] * sx), int(f["y2"] * sy)), (0, 230, 255), 2)
    tag = f"{int(r['no']):03d} {r['image']}" + ("  FAIL" if r["status"] == "fail" else "") + \
          ("  D9" if r["image"] in SUSPECT else "") + ("  M" if f["source"] == "manual" else "")
    cv2.rectangle(vis, (0, 0), (TW, 22), (0, 0, 0), -1)
    cv2.putText(vis, tag, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def sheet(idx_list, path):
    tiles = [thumb(i, auto.iloc[i]) for i in idx_list]
    while len(tiles) % 6:
        tiles.append(np.zeros((TH, TW, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(len(tiles) // 6)])
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


fails = [i for i in range(n) if auto.iloc[i]["status"] == "fail"]
if fails:
    sheet(fails, OUT / "sheet_00_fail.jpg")
    fix_file = OUT / "to_fix.txt"                  # 사람이 적는 파일 → 이미 있으면 실패 번호만 덧붙임
    old = fix_file.read_text().split() if fix_file.exists() else []
    new = [str(int(auto.iloc[i]["no"])) for i in fails if str(int(auto.iloc[i]["no"])) not in old]
    fix_file.write_text("".join(f"{x}\n" for x in old + new))
    print(f"검출 실패 {len(fails)}장 → sheet_00_fail.jpg, 번호를 to_fix.txt에 미리 적어 둠")
for s in range(0, n, 30):
    sheet(list(range(s, min(s + 30, n))), OUT / f"sheet_{s // 30 + 1:02d}.jpg")
print(f"한눈에 보기 그림: outputs/charts/sheet_01~{(n - 1) // 30 + 1:02d}.jpg (30장씩)")

# ── ④ D9 의심 3장: 가림 영역 주변을 크게 잘라 저장 (원인 확인용) ──────────
for name in SUSPECT:
    if name not in set(d["names"]):
        continue
    i = int(np.where(d["names"] == name)[0][0])
    f = final.loc[final["image"] == name].iloc[0]
    lin = load(i)
    scale = 0.9 / np.percentile(lin / (d["illum"][i] / d["illum"][i][1]), 99)
    vis = (to_display(lin, d["illum"][i], scale)[:, :, ::-1] * 255).astype(np.uint8)
    if not pd.isna(f["x1"]):
        x1, y1, x2, y2 = int(f["x1"]), int(f["y1"]), int(f["x2"]), int(f["y2"])
        vis = vis[y1:y2, x1:x2]
    cv2.imwrite(str(OUT / f"suspect_{name.replace('.png', '.jpg')}"), vis)
print("D9 의심 3장: outputs/charts/suspect_*.jpg (회색 칸 위치 · 그림자 확인)")
print("\n다음: 그림에서 노란 상자가 컬러체커를 다 덮지 못한 사진 번호를 outputs/charts/to_fix.txt에 한 줄씩 적고 12 실행")