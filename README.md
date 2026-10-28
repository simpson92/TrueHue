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
- [ ] 정답표 확정 (김하민 검토 후, 기준선 채점 전)
- [ ] 이름 붙이기 규칙 (표준 색 24개 모두 정답)
- [ ] 기준선(보정 없음) 채점
- [ ] 조명 보정 3종(Gray World · White Patch · Shades of Gray) 비교
- [ ] SimpleCube++로 조명 추정 정확도(각도 오차) 평가
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
| `src/00_check_data.py` | Gehler-Shi PNG 개수·dtype·값 범위 확인 | 568장, uint16 선형 이미지 |
| `src/01_inspect_gt.py` | 정답 `.mat` 파일 구조 확인 | 24칸 색 (568×24×3), 조명 정답 (568×3) |
| `src/02_build_index.py` | 사진별 정보를 한 표로 묶고 정답 조명 검산 | `outputs/gehler_index.csv` (568×78), 오차 0 |
| `src/03_inspect_nan.py` | 24칸 빈칸(NaN) 패턴·규칙 조사 | 채점 가능 12,428 / 13,632 |
| `src/04_reference_colors.py` | 컬러체커 24칸 공식 Lab 값, 대표색과 비교 | `reports/patch_reference.csv`, `outputs/patches_preview.png` |
| `src/05_check_simplecube.py` | SimpleCube++ 장수·정답 짝·라벨·블랙 레벨·포화 확인 | 2,234장, 블랙 레벨 2048, 포화 약 15,303 |

```bash
for f in src/0*.py; do echo "== $f"; python "$f"; done
```

## 지금까지의 주요 발견

**Gehler-Shi**
- **카메라 2종**: Canon 1D 86장(`8D5U…`, 블랙 레벨 0) / Canon 5D 482장(`IMG_…`, 블랙 레벨 129).
  파일명 정렬 순서가 정답 파일 순서(1D → 5D)와 일치
- **24칸 평균 색이 이미 제공됨** (블랙 레벨 미적용) → 정답 조명 재계산 오차 0으로 해석 검증
- **빈칸 1,204칸**: 가장 어두운 채널이 20 미만인 칸을 데이터셋이 제외해 둠
  → 채점 가능 **12,428 / 13,632 (91.2%)**. 어두운 색에 몰려 있어 한계로 기록

**정답표**
- "가장 가까운 대표색"으로 자동으로 정하면 5칸이 틀림 (탁한 색 → 회색) → 사람이 근거를 적어 정함
- 확실 13 · 거의 확실 3 · 애매 8칸 (애매한 칸은 두 이름 모두 정답). 현재 **초안**

**SimpleCube++**
- 블랙 레벨 2048, 포화 약 15,303, 큐브 자리는 0으로 칠해져 있음 → 블랙 레벨을 빼기 **전에** 마스크로 제외
- 조명이 하나인 장면만 담음 · 야간 5장뿐 → 조명 그룹은 자연광/인공, 실외/실내로 평가

자세한 내용: [`reports/data_notes.md`](reports/data_notes.md) · 판단 이유: [`reports/decisions.md`](reports/decisions.md)

## 데이터 출처

- Lilong Shi and Brian Funt, "Re-processed Version of the Gehler Color Constancy Dataset of 568 Images,"
  http://www.cs.sfu.ca/~colour/data/
- Peter Gehler et al., "Bayesian Color Constancy Revisited," CVPR 2008
- E. Ershov et al., "The Cube++ Illumination Estimation Dataset," IEEE Access 2020 —
  SimpleCube++, https://github.com/Visillect/CubePlusPlus (CC BY 4.0)
- ColorChecker 공식 값: colour-science 라이브러리 (`ColorChecker24 - Before November 2014`)
- 11색 대표색: xkcd Color Survey