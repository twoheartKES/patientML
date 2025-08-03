"""
모델 학습 및 최적화 관리 모듈

이 모듈은 선형 회귀 모델의 학습 과정을 관리하고,
하이퍼파라미터 최적화 및 교차 검증을 수행하는 기능을 제공합니다.

주요 기능:
- 모델 학습 파이프라인 관리
- 교차 검증을 통한 모델 검증
- 하이퍼파라미터 최적화 (선형 회귀의 경우 제한적)
- 학습 과정 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from .model import DiabetesLinearRegression
from .data_loader import DiabetesDataLoader

class DiabetesModelTrainer:
    """
    당뇨병 선형 회귀 모델 학습을 관리하는 클래스
    
    이 클래스는 데이터 로딩부터 모델 학습, 검증까지의
    전체 파이프라인을 관리합니다.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        모델 트레이너 초기화
        
        매개변수:
        test_size (float): 테스트 세트의 비율 (기본값: 0.2)
        random_state (int): 재현 가능한 결과를 위한 랜덤 시드 (기본값: 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.data_loader = None
        self.model = None
        self.training_results = {}
        self.cv_results = {}
        
        print("당뇨병 모델 트레이너가 초기화되었습니다.")
    
    def setup_data(self, scale_features=True):
        """
        데이터를 로딩하고 전처리합니다.
        
        매개변수:
        scale_features (bool): 특성을 표준화할지 여부 (기본값: True)
        
        반환값:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        print("=== 데이터 설정 시작 ===")
        
        # 데이터 로더 초기화
        self.data_loader = DiabetesDataLoader(
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # 데이터 로딩
        data, target, feature_names = self.data_loader.load_data()
        
        # 데이터 탐색 (시각화 포함)
        self.data_loader.explore_data()
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = self.data_loader.split_data(
            scale_features=scale_features
        )
        
        print("데이터 설정이 완료되었습니다.")
        
        return X_train, X_test, y_train, y_test, feature_names
    
    def train_model(self, model_name="diabetes_linear_regression", 
                   feature_names=None, save_model=True):
        """
        선형 회귀 모델을 학습시킵니다.
        
        매개변수:
        model_name (str): 모델의 이름
        feature_names (list): 특성 이름들의 리스트
        save_model (bool): 학습된 모델을 저장할지 여부
        
        반환값:
        DiabetesLinearRegression: 학습된 모델 인스턴스
        """
        if self.data_loader is None:
            raise ValueError("먼저 setup_data()를 호출하여 데이터를 설정하세요.")
        
        print("=== 모델 학습 시작 ===")
        
        # 모델 초기화
        self.model = DiabetesLinearRegression(model_name)
        
        # 모델 학습
        self.model.train(
            self.data_loader.X_train, 
            self.data_loader.y_train, 
            feature_names=feature_names
        )
        
        # 모델 성능 평가
        evaluation_results = self.model.evaluate(
            self.data_loader.X_test, 
            self.data_loader.y_test
        )
        
        # 학습 결과 저장
        self.training_results = {
            'model_name': model_name,
            'evaluation': evaluation_results,
            'feature_importance': self.model.get_feature_importance()
        }
        
        # 모델 저장
        if save_model:
            self.model.save_model()
        
        print("모델 학습이 완료되었습니다.")
        
        return self.model
    
    def perform_cross_validation(self, cv_folds=5):
        """
        교차 검증을 수행하여 모델의 일반화 성능을 평가합니다.
        
        교차 검증은 데이터를 여러 개의 폴드로 나누어 각 폴드를
        테스트 세트로 사용하여 모델의 안정성을 평가하는 방법입니다.
        
        매개변수:
        cv_folds (int): 교차 검증 폴드 수 (기본값: 5)
        
        반환값:
        dict: 교차 검증 결과
        """
        if self.data_loader is None:
            raise ValueError("먼저 setup_data()를 호출하여 데이터를 설정하세요.")
        
        print(f"\n=== {cv_folds}겹 교차 검증 시작 ===")
        
        # 전체 데이터 준비 (스케일링된 데이터 사용)
        X = self.data_loader.X_train
        y = self.data_loader.y_train
        
        # K-Fold 교차 검증 설정
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # 교차 검증 수행
        cv_scores = cross_val_score(
            self.model.model, 
            X, 
            y, 
            cv=kfold, 
            scoring='r2'
        )
        
        # 결과 저장
        self.cv_results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'cv_folds': cv_folds
        }
        
        # 결과 출력
        print(f"교차 검증 R² 점수:")
        for i, score in enumerate(cv_scores):
            print(f"  Fold {i+1}: {score:.4f}")
        
        print(f"\n평균 R² 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 성능 해석
        self._interpret_cv_results()
        
        return self.cv_results
    
    def _interpret_cv_results(self):
        """
        교차 검증 결과를 해석하고 출력합니다.
        """
        mean_score = self.cv_results['mean_score']
        std_score = self.cv_results['std_score']
        
        print(f"\n=== 교차 검증 결과 해석 ===")
        
        # 점수 해석
        if mean_score >= 0.9:
            performance = "매우 우수"
        elif mean_score >= 0.7:
            performance = "좋음"
        elif mean_score >= 0.5:
            performance = "보통"
        else:
            performance = "개선 필요"
        
        print(f"평균 성능: {performance} ({mean_score:.4f})")
        
        # 표준편차 해석 (안정성)
        if std_score <= 0.05:
            stability = "매우 안정적"
        elif std_score <= 0.1:
            stability = "안정적"
        else:
            stability = "불안정"
        
        print(f"모델 안정성: {stability} (표준편차: {std_score:.4f})")
        
        # 과적합/과소적합 판단
        if hasattr(self, 'training_results'):
            test_score = self.training_results['evaluation']['R² Score']
            score_diff = mean_score - test_score
            
            if abs(score_diff) <= 0.05:
                fit_status = "적절한 적합"
            elif score_diff > 0.05:
                fit_status = "과적합 가능성"
            else:
                fit_status = "과소적합 가능성"
            
            print(f"적합 상태: {fit_status} (차이: {score_diff:.4f})")
    
    def create_learning_visualizations(self):
        """
        학습 결과를 시각화합니다.
        """
        if not self.training_results:
            print("먼저 모델을 학습하세요.")
            return
        
        print("=== 학습 결과 시각화 생성 중 ===")
        
        # 그래프 스타일 설정
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('당뇨병 선형 회귀 모델 학습 결과', fontsize=16, fontweight='bold')
        
        # 1. 실제값 vs 예측값 산점도
        y_test = self.data_loader.y_test
        y_pred = self.model.predict(self.data_loader.X_test)
        
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('실제값')
        axes[0, 0].set_ylabel('예측값')
        axes[0, 0].set_title('실제값 vs 예측값')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 플롯
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('예측값')
        axes[0, 1].set_ylabel('잔차')
        axes[0, 1].set_title('잔차 플롯')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 특성 중요도
        feature_importance = self.training_results['feature_importance']
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
        
        y_pos = np.arange(len(features))
        axes[1, 0].barh(y_pos, importance_values, color='skyblue')
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(features)
        axes[1, 0].set_xlabel('중요도 (계수 절댓값)')
        axes[1, 0].set_title('특성 중요도')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 교차 검증 결과 (있는 경우)
        if self.cv_results:
            cv_scores = self.cv_results['cv_scores']
            fold_numbers = range(1, len(cv_scores) + 1)
            
            axes[1, 1].plot(fold_numbers, cv_scores, 'o-', color='orange', linewidth=2, markersize=8)
            axes[1, 1].axhline(y=cv_scores.mean(), color='r', linestyle='--', 
                              label=f'평균: {cv_scores.mean():.4f}')
            axes[1, 1].set_xlabel('교차 검증 폴드')
            axes[1, 1].set_ylabel('R² 점수')
            axes[1, 1].set_title('교차 검증 결과')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # 성능 지표 요약
            eval_results = self.training_results['evaluation']
            metrics = ['R² Score', 'RMSE', 'MAE']
            values = [eval_results['R² Score'], eval_results['RMSE'], eval_results['MAE']]
            
            bars = axes[1, 1].bar(metrics, values, color=['green', 'red', 'orange'])
            axes[1, 1].set_ylabel('값')
            axes[1, 1].set_title('성능 지표 요약')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 값 표시
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('data/learning_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("시각화 결과가 'data/learning_results.png'에 저장되었습니다.")
    
    def generate_training_report(self):
        """
        학습 결과에 대한 상세한 보고서를 생성합니다.
        
        반환값:
        dict: 학습 보고서
        """
        if not self.training_results:
            print("먼저 모델을 학습하세요.")
            return None
        
        print("=== 학습 보고서 생성 중 ===")
        
        # 데이터 요약
        data_summary = self.data_loader.get_data_summary()
        
        # 모델 정보
        model_info = self.model.get_model_info()
        
        # 평가 결과
        evaluation = self.training_results['evaluation']
        
        # 특성 중요도
        feature_importance = self.training_results['feature_importance']
        
        # 보고서 구성
        report = {
            '프로젝트 정보': {
                '프로젝트명': '당뇨병 환자 선형 회귀 예측',
                '모델 유형': 'Linear Regression',
                '생성일시': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            '데이터 요약': data_summary,
            '모델 정보': model_info,
            '성능 평가': evaluation,
            '특성 중요도': feature_importance
        }
        
        # 교차 검증 결과 추가
        if self.cv_results:
            report['교차 검증'] = self.cv_results
        
        # 보고서 출력
        print("\n" + "="*50)
        print("당뇨병 선형 회귀 모델 학습 보고서")
        print("="*50)
        
        for section, content in report.items():
            print(f"\n[{section}]")
            if isinstance(content, dict):
                for key, value in content.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {content}")
        
        return report
    
    def run_complete_pipeline(self, model_name="diabetes_linear_regression"):
        """
        전체 학습 파이프라인을 실행합니다.
        
        이 메서드는 데이터 설정부터 모델 학습, 검증, 시각화까지
        모든 과정을 자동으로 수행합니다.
        
        매개변수:
        model_name (str): 모델의 이름
        
        반환값:
        tuple: (학습된 모델, 학습 결과, 교차 검증 결과)
        """
        print("="*60)
        print("당뇨병 선형 회귀 모델 전체 파이프라인 시작")
        print("="*60)
        
        # 1. 데이터 설정
        X_train, X_test, y_train, y_test, feature_names = self.setup_data()
        
        # 2. 모델 학습
        model = self.train_model(model_name, feature_names)
        
        # 3. 교차 검증
        cv_results = self.perform_cross_validation()
        
        # 4. 시각화
        self.create_learning_visualizations()
        
        # 5. 보고서 생성
        report = self.generate_training_report()
        
        print("\n" + "="*60)
        print("전체 파이프라인 완료!")
        print("="*60)
        
        return model, self.training_results, cv_results

# 사용 예시
if __name__ == "__main__":
    # 트레이너 인스턴스 생성
    trainer = DiabetesModelTrainer(test_size=0.2, random_state=42)
    
    # 전체 파이프라인 실행
    model, training_results, cv_results = trainer.run_complete_pipeline() 