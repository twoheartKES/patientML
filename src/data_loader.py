"""
당뇨병 환자 데이터 로딩 및 전처리 모듈

이 모듈은 scikit-learn의 당뇨병 데이터셋을 로딩하고,
머신러닝 모델 학습에 적합하도록 전처리하는 기능을 제공합니다.

주요 기능:
- 당뇨병 데이터셋 로딩
- 데이터 탐색 및 기본 통계 정보 출력
- 데이터 분할 (훈련/테스트 세트)
- 특성 스케일링 (필요시)
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class DiabetesDataLoader:
    """
    당뇨병 데이터셋을 로딩하고 전처리하는 클래스
    
    이 클래스는 scikit-learn의 내장 당뇨병 데이터셋을 사용하여
    선형 회귀 모델 학습을 위한 데이터를 준비합니다.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        데이터 로더 초기화
        
        매개변수:
        test_size (float): 테스트 세트의 비율 (기본값: 0.2)
        random_state (int): 재현 가능한 결과를 위한 랜덤 시드 (기본값: 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.data = None
        self.target = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """
        scikit-learn에서 당뇨병 데이터셋을 로딩합니다.
        
        당뇨병 데이터셋은 442명의 당뇨병 환자에 대한 10개의 특성과
        당뇨병 진행 정도를 나타내는 타겟 변수로 구성되어 있습니다.
        
        반환값:
        tuple: (특성 데이터, 타겟 데이터, 특성 이름들)
        """
        print("당뇨병 데이터셋을 로딩 중...")
        
        # scikit-learn의 내장 당뇨병 데이터셋 로딩
        diabetes = load_diabetes()
        
        # 데이터와 타겟 분리
        self.data = diabetes.data
        self.target = diabetes.target
        self.feature_names = diabetes.feature_names
        
        print(f"데이터셋 로딩 완료!")
        print(f"특성 개수: {self.data.shape[1]}")
        print(f"샘플 개수: {self.data.shape[0]}")
        print(f"특성 이름들: {list(self.feature_names)}")
        
        return self.data, self.target, self.feature_names
    
    def explore_data(self):
        """
        데이터의 기본적인 통계 정보와 분포를 탐색합니다.
        
        이 메서드는 데이터의 기본 통계, 상관관계, 분포 등을
        시각화하여 데이터에 대한 이해를 돕습니다.
        """
        if self.data is None:
            print("먼저 데이터를 로딩해주세요!")
            return
        
        print("\n=== 데이터 탐색 ===")
        
        # 데이터프레임 생성
        df = pd.DataFrame(self.data, columns=self.feature_names)
        df['target'] = self.target
        
        # 기본 통계 정보 출력
        print("\n1. 기본 통계 정보:")
        print(df.describe())
        
        # 상관관계 분석
        print("\n2. 타겟과의 상관관계:")
        correlations = df.corr()['target'].sort_values(ascending=False)
        print(correlations)
        
        # 시각화
        self._create_visualizations(df)
        
    def _create_visualizations(self, df):
        """
        데이터 시각화를 생성합니다.
        
        매개변수:
        df (DataFrame): 시각화할 데이터프레임
        """
        # 그래프 스타일 설정
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('당뇨병 데이터셋 분석', fontsize=16, fontweight='bold')
        
        # 1. 타겟 변수 분포
        axes[0, 0].hist(df['target'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('타겟 변수 분포')
        axes[0, 0].set_xlabel('당뇨병 진행 정도')
        axes[0, 0].set_ylabel('빈도')
        
        # 2. 상관관계 히트맵
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0, 1], fmt='.2f')
        axes[0, 1].set_title('특성 간 상관관계')
        
        # 3. 가장 상관관계가 높은 특성과 타겟의 산점도
        top_corr_feature = correlation_matrix['target'].abs().sort_values(ascending=False).index[1]
        axes[1, 0].scatter(df[top_corr_feature], df['target'], alpha=0.6, color='green')
        axes[1, 0].set_xlabel(top_corr_feature)
        axes[1, 0].set_ylabel('타겟')
        axes[1, 0].set_title(f'가장 높은 상관관계: {top_corr_feature}')
        
        # 4. 모든 특성의 분포
        df_features = df.drop('target', axis=1)
        df_features.boxplot(ax=axes[1, 1])
        axes[1, 1].set_title('특성별 분포 (박스플롯)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/diabetes_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("시각화 결과가 'data/diabetes_analysis.png'에 저장되었습니다.")
    
    def split_data(self, scale_features=True):
        """
        데이터를 훈련 세트와 테스트 세트로 분할합니다.
        
        매개변수:
        scale_features (bool): 특성을 표준화할지 여부 (기본값: True)
        
        반환값:
        tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            print("먼저 데이터를 로딩해주세요!")
            return None
        
        print(f"\n데이터 분할 중... (테스트 세트 비율: {self.test_size})")
        
        # 데이터 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, 
            self.target, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        print(f"훈련 세트 크기: {self.X_train.shape}")
        print(f"테스트 세트 크기: {self.X_test.shape}")
        
        # 특성 스케일링 (선택사항)
        if scale_features:
            print("특성 스케일링 적용 중...")
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            print("특성 스케일링 완료!")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_summary(self):
        """
        데이터에 대한 요약 정보를 반환합니다.
        
        반환값:
        dict: 데이터 요약 정보
        """
        if self.data is None:
            return {"error": "데이터가 로딩되지 않았습니다."}
        
        summary = {
            "전체 샘플 수": self.data.shape[0],
            "특성 수": self.data.shape[1],
            "특성 이름": list(self.feature_names),
            "타겟 범위": f"{self.target.min():.2f} ~ {self.target.max():.2f}",
            "타겟 평균": f"{self.target.mean():.2f}",
            "타겟 표준편차": f"{self.target.std():.2f}"
        }
        
        if self.X_train is not None:
            summary.update({
                "훈련 세트 크기": self.X_train.shape[0],
                "테스트 세트 크기": self.X_test.shape[0]
            })
        
        return summary

# 사용 예시
if __name__ == "__main__":
    # 데이터 로더 인스턴스 생성
    data_loader = DiabetesDataLoader(test_size=0.2, random_state=42)
    
    # 데이터 로딩
    data, target, feature_names = data_loader.load_data()
    
    # 데이터 탐색
    data_loader.explore_data()
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = data_loader.split_data(scale_features=True)
    
    # 데이터 요약 출력
    summary = data_loader.get_data_summary()
    print("\n=== 데이터 요약 ===")
    for key, value in summary.items():
        print(f"{key}: {value}") 