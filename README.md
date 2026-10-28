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
- [ ] 568장 컬러체커 위치 적용 + 사람 검수
- [ ] 정답표·이름 규칙 확정 (김하민 검토 후, 기준선 채점 전)
- [ ] 조명 추정 3종(Gray World · White Patch · Shades of Gray) 구현
- [ ] SimpleCube++로 조명 추정 정확도(각도 오차) 평가
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

`src/common.py`(경로·규칙 숫자·데이터 읽기, 사진은 RGB로)와 `src/pipeline.py`(조명 → 노출 → 행렬 → Lab → 이름)는
다른 스크립트가 불러다 쓰는 부품이라 직접 실행하지 않는다 (`python src/common.py`는 채널 순서 자기 점검).
`src/namer.py`도 부품이다. 07~10은 02가 만든 `outputs/gehler_index.csv`가 필요하다.

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
  → 표시해 두고 최종 결과를 "568장 전체" · "3장 제외" 두 가지로 보고

**정답표**
- "가장 가까운 대표색"으로 자동으로 정하면 5칸이 틀림 (탁한 색 → 회색) → 사람이 근거를 적어 정함
- 확실 13 · 거의 확실 3 · 애매 8칸 (애매한 칸은 두 이름 모두 정답). 현재 **초안**

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