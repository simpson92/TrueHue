# TrueHue — 조명에 속지 않는 색 이름 감별기

> 조명 색을 먼저 추정·제거한 뒤 색 이름을 붙여,
> 형광등·백열등·야간 조명에서도 색 이름이 틀리지 않게 만드는 감별기

기존 색 이름 앱은 찍힌 픽셀 색을 그대로 이름으로 바꿔서, 조명이 바뀌면 같은 물체도 다른 이름이 나온다.
색각이상이 있는 사용자는 이 틀린 답을 스스로 알아챌 수 없다.
이 프로젝트는 **조명 보정 단계를 하나 추가**하고, 컬러체커가 찍힌 공개 데이터셋으로
기존 방식(기준선) 대비 정답률이 얼마나 오르는지 **숫자로** 보인다.

## 진행 현황

- [x] 레포·환경 구성
- [x] Gehler-Shi 데이터 받기·확인 (568장)
- [x] 정답 파일 해석·검산, 사진별 인덱스 표 생성
- [x] 빈칸(NaN) 패턴 조사 → 채점 가능 문제 12,428개 확정
- [x] 컬러체커 24칸 공식 Lab 값 확보, 정답표 **초안** 작성
- [x] SimpleCube++ 받기·확인 (2,234장, Cube+ 대체)
- [x] 색 이름 규칙 **초안** (표준 색 24개 모두 정답)
- [x] 보정용 / 봉인 평가용 분할 고정
- [x] 카메라별 색 변환 행렬 (보정용 사진으로)
- [x] 중간 점검 · 공통 모듈 정리 (RGB 읽기, 처리 흐름 한 곳으로)
- [x] 컬러체커 위치 찾는 방법 결정 (정답 색으로 찾기, 처음 보는 30장 중 29장)
- [x] 중간 점검 데이터 확인 (1D 채널 순서 · 흰색 빈칸 · 정답 조명 의심 사진)
- [x] 568장 컬러체커 위치 적용 + 사람 검수 (자동 528장 · 손으로 40장 → `reports/chart_boxes.csv`)
- [x] 정답표·이름 규칙 확정 (2026-10-29 의논, 초안 그대로)
- [x] 채점 · 보고 규칙 확정 (이름 일관성 지표 · 조명 그룹 · 오차 범위 · 주 결과, D10) · 조명 추정 설정 확정 (D11)
- [x] 조명 추정 부품(`illuminant.py`) · Gray World · White Patch 구현
- [x] SimpleCube++ 보정용(train) 각도 오차 — 중앙값 Gray World 1.91° · White Patch 2.44°
- [x] Shades of Gray p 정하기 — SimpleCube++ · Gehler 모두 **p = 3** (`reports/illum_params.json`)
- [x] Gehler 보정용 각도 오차 (가림 영역 적용) — 중앙값 Gray World 3.72° · Shades of Gray 2.51° · White Patch 6.93°
- [ ] 기준선 · 상한선 · 보정 3종 색 이름 채점 (보정용으로 먼저 확인)
- [ ] SimpleCube++ test로 조명 추정 정확도(각도 오차) 최종 평가
- [ ] 기준선 · 보정 3종 · 상한선 색 이름 채점
- [ ] 최종 1회 평가 · 웹 화면(FastAPI) · 리포트
- [ ] (확장) 섞인 조명 · 야간 한계 분석

## 폴더 구조

```
TrueHue/
├── data/          # 데이터셋 원본 (gitignore, 아래 '데이터 준비'로 직접 받기)
│   ├── gehler_shi/    # png/ 568장 + 정답 .mat
│   └── simplecube/    # SimpleCube++/ train · test · auxiliary
├── splits/        # 보정용 / 봉인 평가용 사진 목록
├── src/           # 전처리·보정·채점 스크립트 (번호 순서대로 실행)
├── notebooks/     # 중간 결과 확인용 Jupyter
├── reports/       # 데이터 노트, 결정 기록, 정답표, 비교 리포트
├── app/           # 웹 화면 (예정)
└── outputs/       # 실행 결과물 (gitignore, 스크립트로 다시 생성)
```

## 환경 설정 (WSL Ubuntu 기준)

**Python 3.14 기준.** `requirements.txt`의 버전(numpy 2.5, pandas 3.1, opencv 4.14 등)은 최신 파이썬에서만 설치된다.
`python3 --version`으로 먼저 확인할 것.

```bash
git clone https://github.com/simpson92/TrueHue.git
cd TrueHue
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 데이터 준비

### 1. Gehler-Shi (SFU 재처리판, 약 4.2GB) — 색 이름 채점 (메인)

```bash
mkdir -p data/gehler_shi && cd data/gehler_shi

# 받기 (aria2c가 있으면 더 빠름: aria2c -x 8 -s 8 -j 5 -c <주소들>)
wget -c \
  http://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon1d.zip \
  http://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon5d_1.zip \
  http://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon5d_2.zip \
  http://www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon5d_3.zip \
  https://www2.cs.sfu.ca/~colour/data/shi_gehler/groundtruth_568.zip

# 압축 검사 → 풀기 → 맥 찌꺼기 삭제
for f in *.zip; do unzip -tq "$f"; done
for f in *.zip; do unzip -q "$f"; done
rm -rf __MACOSX

# PNG를 png/ 폴더 하나로 모으기
mkdir -p png
find . -name "*.png" -not -path "./png/*" -exec mv {} png/ \;
ls png | wc -l          # 568 확인
rm -rf cs *.zip
cd ../..
```

### 2. SimpleCube++ (약 2GB) — 조명 추정 정확도 평가 (보조)

원래 계획한 Cube+는 배포 링크가 만료되어, 같은 카메라·기준 물체로 만든 후속판으로 대체했다
(이유: [`reports/decisions.md`](reports/decisions.md) D1).

```bash
mkdir -p data/simplecube && cd data/simplecube
wget -c "https://zenodo.org/record/4153431/files/SimpleCube++.zip"
unzip -tq SimpleCube++.zip && unzip -q SimpleCube++.zip
find SimpleCube++/train SimpleCube++/test -name "*.png" | wc -l    # 2234 확인
rm SimpleCube++.zip
cd ../..
```

## 실행 순서

| 스크립트 | 하는 일 | 결과 |
|---|---|---|
| `src/00_check_data.py` | Gehler-Shi · SimpleCube++ PNG 개수·dtype·값 범위 확인 | 568장 · 1,772장 · 462장, uint16 선형 이미지 |
| `src/01_inspect_gt.py` | 정답 `.mat` 파일 구조 확인 | 24칸 색 (568×24×3), 조명 정답 (568×3) |
| `src/02_build_index.py` | 사진별 정보를 한 표로 묶고 정답 조명 검산 | `outputs/gehler_index.csv` (568×78), 오차 0 |
| `src/03_inspect_nan.py` | 24칸 빈칸(NaN) 패턴·규칙 조사 | 채점 가능 12,428 / 13,632 |
| `src/04_reference_colors.py` | 컬러체커 24칸 공식 Lab 값, 대표색과 비교 | `reports/patch_reference.csv`, `outputs/patches_preview.png` |
| `src/05_check_simplecube.py` | SimpleCube++ 장수·정답 짝·라벨·블랙 레벨·포화 확인 | 2,234장, 블랙 레벨 2048, 포화 약 15,303 |
| `src/06_test_namer.py` | 색 이름 규칙(`src/namer.py`)이 표준 색 24개를 정답표대로 맞히는지 검증 | 24/24 정답, `outputs/namer_hue_chroma.png` |
| `src/07_make_splits.py` | 보정용 1/3 · 봉인 평가용 2/3 분할 (이미 있으면 덮어쓰지 않음) | `splits/*.txt`, 보정용 190장 / 평가용 378장 |
| `src/08_fit_color_matrix.py` | 보정용 사진으로 카메라별 3×3 색 변환 행렬 맞추기 | `reports/color_matrix.json`, ΔE 중앙값 5D 2.55 · 1D 5.03 |
| `src/09_pilot_chart_detect.py` | (시험) 컬러체커 자동 검출 방법 비교 — 정답 색으로 찾기 3차, 처음 보는 30장 중 29장 | `outputs/chart_pilot*/`, 결과는 decisions.md D8 |
| `src/10_check_camera_white.py` | (점검) 1D 채널 순서 · 흰색 빈칸 원인 · SFU 정답 조명 의심 사진 | 순서 맞음, 9/9장 포화, 의심 3장 (data_notes · D9) |
| `src/11_detect_charts.py` | 568장 컬러체커 자동 검출(09 3차 방식) → 가림 영역(상자 긴 변 25% 여유) · 30장씩 한눈에 보기 그림 | `outputs/charts/sheet_*.jpg`, `reports/chart_boxes.csv`(처음 한 번만 만듦) |
| `src/12_fix_charts.py` | (사람) `outputs/charts/to_fix.txt`에 적은 사진만 열어 마우스로 가림 영역 다시 지정 | `reports/chart_boxes.csv` 수정(source = manual), `sheet_fixed.jpg` |
| `src/13_check_suspect_matrix.py` | (점검) 의심 사진 8D5U5541을 빼고 1D 행렬을 다시 맞춰 ΔE 변화 확인 (D9, 처리 흐름은 안 바꿈) | 같은 28장 ΔE 중앙값 4.83 → 4.73, 평균 5.85 → 5.84 |
| `src/14_illum_simplecube.py` | Gray World · White Patch를 SimpleCube++ train 1,772장에 돌려 각도 오차 (test는 열지 않음) | `outputs/illum/simplecube_train.csv`, 중앙값 1.91° · 2.44° |
| `src/15_sog_simplecube.py` | Shades of Gray p = 1~12를 SimpleCube++ train에서 비교 → 중앙값 최소 p (D11) | `reports/illum_params.json`(simplecube), p = 3 · 1.45° |
| `src/16_illum_gehler.py` | Gehler 보정용 190장에서 가림 영역을 빼고 p = 1~12 비교 → 색 이름 채점에 쓸 p (D11) | `reports/illum_params.json`(gehler), p = 3 · 2.51° |

`src/common.py`(경로·규칙 숫자·데이터 읽기, 사진은 RGB로)와 `src/pipeline.py`(조명 → 노출 → 행렬 → Lab → 이름)는
다른 스크립트가 불러다 쓰는 부품이라 직접 실행하지 않는다 (`python src/common.py`는 채널 순서 자기 점검).
`src/namer.py`(이름 규칙), `src/chart.py`(컬러체커 검출 · 가림 영역), `src/illuminant.py`(조명 추정: p=1 Gray World · p=∞ White Patch · 그 사이 Shades of Gray)도 부품이다.
07~13 · 16은 02가 만든 `outputs/gehler_index.csv`가 필요하다. Gehler 사진으로 조명을 추정할 때는 `common.load_gehler_image`(포화 + 컬러체커 가림 영역 제외)로만 읽는다.
12는 사람이 마우스로 고치는 단계라 `to_fix.txt`에 적은 사진이 없으면 아무것도 하지 않는다.

```bash
for f in src/[0-9]*.py; do echo "== $f"; python "$f"; done    # 번호가 붙은 스크립트만 순서대로
```

## 지금까지의 주요 발견

**Gehler-Shi**
- **카메라 2종**: Canon 1D 86장(`8D5U…`, 블랙 레벨 0) / Canon 5D 482장(`IMG_…`, 블랙 레벨 129).
  파일명 정렬 순서가 정답 파일 순서(1D → 5D)와 일치
- **24칸 평균 색이 이미 제공됨** (블랙 레벨 미적용) → 정답 조명 재계산 오차 0으로 해석 검증
- **빈칸 1,204칸**: 어둡거나(가장 어두운 채널 < 20) 포화된 칸을 데이터셋이 제외해 둠
  → 채점 가능 **12,428 / 13,632 (91.2%)**. 대부분 어두운 색이라 한계로 기록. 흰색 빈칸 9장은 모두 포화
- **정답 조명 의심 3장** (모두 1D): R·G·B가 서로 다른 회색 칸에서 옴 (Hemrit et al. 2018이 지적한 문제)
  → 표시해 두고 최종 결과를 두 가지로 보고. 평가용에는 2장이라 "평가용 378장" · "의심 2장 제외 376장"
  → 보정용에 있는 8D5U5541이 1D 행렬에 준 영향은 작음 (13번: 같은 28장 ΔE 중앙값 4.83 → 4.73, 평균 5.85 → 5.84)
- **컬러체커 위치**: 원본 좌표가 없어 직접 만듦. 자동 검출이 맞은 사진 528 / 568장(93.0%),
  나머지 40장(검출 실패 2 · 일부만 덮음 · 다른 곳을 덮음 · 그림에서 상자가 안 보임)은 사람이 마우스로 다시 지정 → `reports/chart_boxes.csv`

**정답표**
- "가장 가까운 대표색"으로 자동으로 정하면 5칸이 틀림 (탁한 색 → 회색) → 사람이 근거를 적어 정함
- 확실 13 · 거의 확실 3 · 애매 8칸 (애매한 칸은 두 이름 모두 정답). 2026-10-29 **확정**

**SimpleCube++**
- 블랙 레벨 2048, 포화 약 15,303, 큐브 자리는 0으로 칠해져 있음 → 블랙 레벨을 빼기 **전에** 마스크로 제외
- 조명이 하나인 장면만 담음 · 야간 5장뿐 → 조명 그룹은 자연광/인공, 실외/실내로 평가

**색 이름 규칙 · 분할 · 색 변환**
- 규칙: 채도 < 10이면 무채색(밝기로 검정·회색·흰색), 아니면 색상 각도 7구간 + 어두운 주황은 갈색.
  기준값은 표준 색만 보고 정함 → 24칸 모두 정답. 회색 칸은 채도 여유가 약 9뿐이라 색 있는 조명에 약할 것으로 예상
- 분할: 정할 값이 약 19개(색 변환 9 × 카메라 2 + Shades of Gray 1)뿐이라 보정용 1/3(4,188칸, 행렬이 가장 적은 1D도 683칸)로 충분, 평가용 2/3(8,240칸)로 최종 점수의 오차 범위를 줄임
- 색 변환: 진한 색(빨강·파랑·보라·귤색)에서 오차가 큼 → 조명을 완벽히 맞힌 상한선도 100%가 아닐 수 있음
- 1D 오차가 5D의 약 2배: 채널 순서 문제가 아니라 센서 특성 → 리포트에 카메라별 점수를 따로 표기
- 비교 설계: 기준선(평균 조명 고정) · 보정 3종 · 상한선(정답 조명)을 같은 색 변환·같은 이름 규칙으로 비교

**조명 추정 첫 결과 (보정용 SimpleCube++ train 1,772장, 최종 숫자 아님)**
- 각도 오차 중앙값 Gray World **1.91°** · White Patch **2.44°** (평균 3.10° · 5.23°, 나쁜 25% 평균 7.63° · 14.43°, 최대 43° · 53°)
  → 몇몇 사진에서 크게 틀리는 긴 꼬리가 있어 중앙값을 주로 본다
- White Patch는 실내 · 인공조명에서 크게 약함: 인공조명(89장) 6.45° vs Gray World 1.48°, 실내(310장) 4.12° vs 1.47°.
  실외(1,363장)는 둘 다 2.09°
- 사진별로는 Gray World가 58%에서 더 잘 맞음 → 두 방법의 약점이 달라, 사이 값인 Shades of Gray를 시험할 이유가 된다
- **Shades of Gray (p = 3)**: SimpleCube++ train 중앙값 **1.45°** (Gray World 1.91° · White Patch 2.44°).
  인공조명(89장)에서는 1.59°로 Gray World(1.48°)와 비슷하고 White Patch(6.45°)보다 훨씬 나음
- **Gehler 보정용 190장 (가림 영역 적용)**: 중앙값 Gray World 3.72° · **Shades of Gray(p = 3) 2.51°** · White Patch 6.93°.
  두 데이터셋 모두 p = 3이 골라졌고, p = 3~5 사이는 중앙값 차이가 0.05° 안팎이라 p 선택에 민감하지 않음
- Gehler가 SimpleCube++보다 어려움 (2.51° vs 1.45°): 조명이 섞인 장면, 큰 가림 영역(쓸 수 있는 픽셀이 22%까지 줄어든 사진), 정답 조명 문제(D9)가 원인 후보

**채점 규칙 (2026-10-29 확정, 채점 전에 고정)**
- 지표: 정답률(애매한 칸은 두 이름 모두 정답) + 보조 지표 **이름 일관성**(표준 색에 붙는 이름으로 불린 비율 — 조명 때문에 이름이 흔들리는지)
- **주 결과**: 보정 3종 각각 vs 기준선, 평가용 전체 24칸 정답률 차이. 나머지는 보조 결과
- 오차 범위: 평가용 사진 단위 부트스트랩(2,000회, 95%, 카메라별), 방법끼리는 같은 사진으로 짝지어 비교
- Gehler 조명 그룹: 정답 조명 색온도로 나눔 (4,000K 미만 따뜻함 / 4,000~6,500K 주광 / 6,500K 초과 푸름)
- 예측: 회색 칸이 가장 많이 틀릴 것 (조명 추정 1° 오차당 회색 칸 채도 약 2.4, 무채색 기준 10 ≈ 약 4°)
- 조명 추정: 기준선 평균 조명은 카메라별 · Shades of Gray p는 데이터셋별 보정용으로 · White Patch는 채널별 최댓값

자세한 내용: [`reports/data_notes.md`](reports/data_notes.md) · 판단 이유: [`reports/decisions.md`](reports/decisions.md)

## 데이터 출처

- Lilong Shi and Brian Funt, "Re-processed Version of the Gehler Color Constancy Dataset of 568 Images,"
  http://www.cs.sfu.ca/~colour/data/
- Peter Gehler et al., "Bayesian Color Constancy Revisited," CVPR 2008
- G. Hemrit et al., "Rehabilitating the ColorChecker Dataset for Illuminant Estimation," CIC 2018 (arXiv 1805.12262)
- E. Ershov et al., "The Cube++ Illumination Estimation Dataset," IEEE Access 2020 —
  SimpleCube++, https://github.com/Visillect/CubePlusPlus (CC BY 4.0)
- ColorChecker 공식 값: colour-science 라이브러리 (`ColorChecker24 - Before November 2014`)
- 11색 대표색: xkcd Color Survey