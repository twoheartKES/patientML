# 당뇨병 환자 선형 회귀 머신러닝 프로젝트

## 프로젝트 개요
이 프로젝트는 scikit-learn의 당뇨병 데이터셋을 사용하여 선형 회귀(Linear Regression) 모델을 학습시키는 머신러닝 프로젝트입니다.

## 프로젝트 구조
```
patientML/
├── data/                   # 데이터 저장 폴더
├── models/                 # 학습된 모델 저장 폴더
├── notebooks/              # Jupyter 노트북 파일들
├── src/                    # 소스 코드
│   ├── __init__.py
│   ├── data_loader.py      # 데이터 로딩 및 전처리
│   ├── model.py           # 선형 회귀 모델 정의
│   ├── trainer.py         # 모델 학습
│   └── evaluator.py       # 모델 평가
├── main.py                # 메인 실행 파일
├── requirements.txt       # 필요한 라이브러리 목록
└── README.md             # 프로젝트 설명서
```

## 설치 및 실행 방법

### 1. 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. 필요한 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 3. 프로젝트 실행
```bash
python main.py
```

## 주요 기능
- 당뇨병 데이터셋 로딩 및 전처리
- 선형 회귀 모델 학습
- 모델 성능 평가 및 시각화
- 학습된 모델 저장 및 로딩

## 사용된 기술
- **Python 3.8+**
- **scikit-learn**: 머신러닝 라이브러리
- **pandas**: 데이터 처리
- **numpy**: 수치 계산
- **matplotlib/seaborn**: 데이터 시각화 