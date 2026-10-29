# src/12_fix_charts.py
# 목적: 11의 한눈에 보기 그림에서 "노란 상자가 컬러체커를 다 덮지 못한" 사진만 골라, 마우스로 가림 영역을 다시 지정한다
#   방법: 교과 9에서 쓴 cv2.selectROI (마우스로 드래그 → Enter/Space로 확정, c로 건너뛰기)
#   - 고칠 사진 번호는 outputs/charts/to_fix.txt에 한 줄에 하나씩 적는다 (번호 001~568 또는 파일 이름)
#   - 드래그한 사각형이 그대로 "가림 영역"이 된다 → 컬러체커 전체 + 다리 조금까지 넉넉하게 드래그
#   - 한 장 고칠 때마다 reports/chart_boxes.csv에 바로 저장 (source = manual) → 중간에 멈춰도 됨
#   - 다 끝나면 고친 사진만 모은 확인 그림 outputs/charts/sheet_fixed.jpg를 만든다
import cv2
import numpy as np
import pandas as pd
from common import ROOT, GEHLER_DIR, GEHLER_BLACK, read_rgb, load_gehler_index
from chart import to_display

OUT = ROOT / "outputs" / "charts"
FINAL = ROOT / "reports" / "chart_boxes.csv"
FIX = OUT / "to_fix.txt"
SCREEN_W, SCREEN_H = 1500, 950                     # 창 크기 (화면에 맞게 바꿔도 됨)

final = pd.read_csv(FINAL, dtype={"note": "string"})
final["note"] = final["note"].fillna("")
BOX = ["x1", "y1", "x2", "y2"]


def save():
    """좌표를 정수로(빈칸은 비워 둔 채) 저장 — 240.0처럼 소수점이 붙지 않게"""
    out = final.copy()
    out[BOX] = out[BOX].round().astype("Int64")
    out.to_csv(FINAL, index=False)
d = load_gehler_index()
pos = {name: i for i, name in enumerate(d["names"])}

# to_fix.txt → 사진 이름 목록 (번호 · 파일 이름 둘 다 받음, 빈 줄 · # 주석 무시)
targets, again = [], set()
for tok in FIX.read_text().split("\n") if FIX.exists() else []:
    tok = tok.split("#")[0].strip()
    if not tok:
        continue
    force = tok.endswith("!")                       # 이미 고친 사진을 다시 고치려면 번호 뒤에 !
    tok = tok.rstrip("!").strip()
    if tok.isdigit():
        hit = final.loc[final["no"] == int(tok), "image"]
    else:
        hit = final.loc[final["image"] == (tok if tok.endswith(".png") else tok + ".png"), "image"]
    if len(hit):
        targets.append(hit.iloc[0])
        if force:
            again.add(hit.iloc[0])
    else:
        print(f"⚠️ 목록에 없는 번호/이름: {tok}")
targets = list(dict.fromkeys(targets))             # 중복 제거 (순서 유지)
done = [t for t in targets if t not in again and
        final.loc[final["image"] == t, "source"].iloc[0] == "manual"]
if done:
    print(f"이미 손으로 고친 {len(done)}장은 건너뜀 {done} (다시 하려면 to_fix.txt에서 번호 뒤에 ! 붙이기)")
targets = [t for t in targets if t not in done]
print(f"고칠 사진 {len(targets)}장 — 드래그 후 Enter(또는 Space) 확정, c = 건너뛰기, 창을 닫으려면 끝까지 진행")


def display(i):
    """사진 i → 화면에 맞게 줄인 확인용 그림(BGR)과 줄인 배율"""
    raw = read_rgb(GEHLER_DIR / "png" / d["names"][i]).astype(np.float64)
    lin = np.clip(raw - GEHLER_BLACK[d["cams"][i]], 0, None)
    s = min(SCREEN_W / lin.shape[1], SCREEN_H / lin.shape[0], 1.0)
    small = cv2.resize(lin.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    scale = 0.9 / np.percentile(small / (d["illum"][i] / d["illum"][i][1]), 99)
    vis = (to_display(small, d["illum"][i], scale)[:, :, ::-1] * 255).astype(np.uint8).copy()
    return vis, s


fixed = []
for k, name in enumerate(targets, 1):
    i = pos[name]
    row = final.index[final["image"] == name][0]
    vis, s = display(i)
    if not pd.isna(final.at[row, "x1"]):            # 지금 가림 영역(노랑)을 참고로 보여 줌
        x1, y1, x2, y2 = (final.loc[row, ["x1", "y1", "x2", "y2"]] * s).astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 230, 255), 2)
    title = f"[{k}/{len(targets)}] {int(final.at[row, 'no']):03d} {name} - drag chart, Enter=OK, c=skip"
    x, y, w, h = cv2.selectROI(title, vis, showCrosshair=False)
    cv2.destroyWindow(title)
    if w == 0 or h == 0:                            # c 또는 빈 선택 → 그대로 둠
        print(f"  {name}: 건너뜀")
        continue
    # 줄인 그림의 좌표 → 원본 좌표
    final.loc[row, BOX] = [int(x / s), int(y / s), int((x + w) / s), int((y + h) / s)]
    final.at[row, "status"] = "ok"
    final.at[row, "source"] = "manual"
    save()                                          # 한 장마다 저장
    fixed.append(i)
    print(f"  {name}: 저장 {[int(v) for v in final.loc[row, BOX]]}")

# ── 고친 사진 확인 그림 ─────────────────────────────────────────────
if fixed:
    tiles = []
    for i in fixed:
        vis, s = display(i)
        r = final.loc[final["image"] == d["names"][i]].iloc[0]
        cv2.rectangle(vis, (int(r["x1"] * s), int(r["y1"] * s)), (int(r["x2"] * s), int(r["y2"] * s)),
                      (0, 230, 255), 3)
        tiles.append(cv2.resize(vis, (480, 320)))
    while len(tiles) % 6:
        tiles.append(np.zeros((320, 480, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(len(tiles) // 6)])
    cv2.imwrite(str(OUT / "sheet_fixed.jpg"), grid)
    print(f"\n고친 {len(fixed)}장 확인 그림: outputs/charts/sheet_fixed.jpg")

left = final[final["source"] == "none"]
print(f"reports/chart_boxes.csv — auto {(final.source == 'auto').sum()} · manual {(final.source == 'manual').sum()}"
      f" · 아직 없음 {len(left)}장" + (f" {left['image'].tolist()}" if len(left) else ""))