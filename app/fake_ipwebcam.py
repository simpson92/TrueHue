# app/fake_ipwebcam.py
# 목적: 폰 없이 ③ 실시간 카메라 모드를 시험하는 가짜 IP Webcam
#   - IP Webcam과 같은 주소(/shot.jpg)로 그림 한 장을 준다
#   - 그림 = 회색 배경 위 컬러체커 24칸(표준 색). 조명 색이 4초마다 주광 → 따뜻한 → 푸른 순서로 바뀐다
#     (폰의 화이트밸런스를 "고정"한 것처럼 조명 색이 그대로 찍힘 → 보정 전 이름은 틀리고, 보정 후 이름은 그대로여야 정상)
#   실행 (레포 폴더에서):  python app/fake_ipwebcam.py            → 웹 화면 ③에 주소 127.0.0.1:8080
#                          python app/fake_ipwebcam.py --port 8081
import argparse
import io
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import colour
from PIL import Image

# 조명 색 (선형 RGB 이득, 큰 값 = 1). 따뜻한 ≈ 백열등, 푸른 ≈ 그늘
LIGHTS = [("주광", [1.00, 1.00, 1.00]), ("따뜻한 조명", [1.00, 0.72, 0.42]), ("푸른 조명", [0.70, 0.86, 1.00])]
PERIOD = 4.0                                    # 초마다 조명 바꿈
W, H = 640, 480


def chart_linear_srgb():
    """컬러체커 24칸 표준 색 → 선형 sRGB (24, 3)"""
    cc = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
    xyz = colour.xyY_to_XYZ(np.array(list(cc.data.values())))
    return np.clip(colour.XYZ_to_RGB(xyz, "sRGB", illuminant=cc.illuminant,
                                     chromatic_adaptation_transform="Bradford"), 0, 1)


def make_scene():
    """선형 sRGB 장면 (H, W, 3): 18% 회색 배경 + 4×6 칸"""
    img = np.full((H, W, 3), 0.18)
    rgb = chart_linear_srgb()
    cw, ch, gap, x0, y0 = 80, 80, 12, 44, 52
    for k in range(24):
        r, c = divmod(k, 6)
        x, y = x0 + c * (cw + gap), y0 + r * (ch + gap)
        img[y:y + ch, x:x + cw] = rgb[k]
    return img


SCENE = make_scene()
RNG = np.random.default_rng(42)


def frame_jpeg():
    name, gain = LIGHTS[int(time.time() // PERIOD) % len(LIGHTS)]
    g = np.array(gain)
    lin = SCENE * g * (0.75 / (g @ [0.2126, 0.7152, 0.0722]))   # 폰 자동 노출처럼 밝기는 맞추고, 색은 그대로
    lin = lin + RNG.normal(0, 0.004, lin.shape)              # 센서 잡음 조금
    srgb = colour.cctf_encoding(np.clip(lin, 0, 1), function="sRGB")
    buf = io.BytesIO()
    Image.fromarray((srgb * 255 + 0.5).astype(np.uint8)).save(buf, format="JPEG", quality=85)
    return buf.getvalue(), name


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] != "/shot.jpg":
            self.send_error(404, "가짜 IP Webcam은 /shot.jpg만 있어요")
            return
        data, _ = frame_jpeg()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):                            # 요청마다 줄 찍지 않음
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    port = ap.parse_args().port
    print(f"가짜 IP Webcam: http://127.0.0.1:{port}/shot.jpg (조명 {PERIOD:.0f}초마다 주광 → 따뜻한 → 푸른). 끄려면 Ctrl+C")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()