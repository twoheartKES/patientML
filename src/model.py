"""
선형 회귀 모델 정의 및 관리 모듈

이 모듈은 당뇨병 환자 데이터를 예측하기 위한 선형 회귀 모델을 정의하고,
모델의 성능을 평가하는 기능을 제공합니다.

주요 기능:
- 선형 회귀 모델 정의
- 모델 학습 및 예측
- 모델 성능 평가
- 모델 저장 및 로딩
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from datetime import datetime

class DiabetesLinearRegression:
    """
    당뇨병 환자 데이터를 위한 선형 회귀 모델 클래스
    
    이 클래스는 scikit-learn의 LinearRegression을 래핑하여
    당뇨병 진행 정도를 예측하는 모델을 제공합니다.
    """
    
    def __init__(self, model_name="diabetes_linear_regression"):
        """
        선형 회귀 모델 초기화
        
        매개변수:
        model_name (str): 모델의 이름 (기본값: "diabetes_linear_regression")
        """
        self.model_name = model_name
        self.model = LinearRegression()
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        
        print(f"선형 회귀 모델 '{model_name}'이 초기화되었습니다.")
    
    def train(self, X_train, y_train, feature_names=None):
        """
        선형 회귀 모델을 훈련 데이터로 학습시킵니다.
        
        선형 회귀는 다음과 같은 수식으로 예측을 수행합니다:
        y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
        
        여기서:
        - y: 예측할 타겟 변수 (당뇨병 진행 정도)
        - β₀: 절편 (intercept)
        - βᵢ: i번째 특성의 계수 (coefficient)
        - xᵢ: i번째 특성 값
        
        매개변수:
        X_train (array-like): 훈련 특성 데이터
        y_train (array-like): 훈련 타겟 데이터
        feature_names (list): 특성 이름들의 리스트
        
        반환값:
        self: 학습된 모델 인스턴스
        """
        print(f"\n=== 모델 학습 시작 ===")
        print(f"훈련 데이터 크기: {X_train.shape}")
        
        # 특성 이름 저장
        if feature_names is not None:
            self.feature_names = feature_names
        
        # 모델 학습 시작 시간 기록
        start_time = datetime.now()
        
        # 선형 회귀 모델 학습
        self.model.fit(X_train, y_train)
        
        # 학습 완료 시간 기록
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        # 학습 결과 저장
        self.is_trained = True
        self.training_history = {
            'training_time': training_time,
            'training_samples': X_train.shape[0],
            'features_count': X_train.shape[1],
            'intercept': self.model.intercept_,
            'coefficients': self.model.coef_
        }
        
        print(f"모델 학습 완료! (소요 시간: {training_time:.2f}초)")
        print(f"절편 (β₀): {self.model.intercept_:.4f}")
        print(f"계수 개수: {len(self.model.coef_)}")
        
        # 특성별 계수 출력
        if self.feature_names:
            print("\n특성별 계수:")
            for i, (feature, coef) in enumerate(zip(self.feature_names, self.model.coef_)):
                print(f"  {feature}: {coef:.4f}")
        
        return self
    
    def predict(self, X):
        """
        학습된 모델을 사용하여 새로운 데이터에 대한 예측을 수행합니다.
        
        매개변수:
        X (array-like): 예측할 특성 데이터
        
        반환값:
        array: 예측 결과
        """
        if not self.is_trained:
            raise ValueError("모델이 아직 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        모델의 성능을 평가합니다.
        
        평가 지표:
        - R² 점수 (결정 계수): 모델이 설명하는 분산의 비율 (1에 가까울수록 좋음)
        - MSE (평균 제곱 오차): 예측 오차의 제곱 평균 (낮을수록 좋음)
        - RMSE (평균 제곱근 오차): MSE의 제곱근 (낮을수록 좋음)
        - MAE (평균 절대 오차): 예측 오차의 절댓값 평균 (낮을수록 좋음)
        
        매개변수:
        X_test (array-like): 테스트 특성 데이터
        y_test (array-like): 테스트 타겟 데이터
        
        반환값:
        dict: 평가 지표들을 담은 딕셔너리
        """
        if not self.is_trained:
            raise ValueError("모델이 아직 학습되지 않았습니다.")
        
        print(f"\n=== 모델 성능 평가 ===")
        print(f"테스트 데이터 크기: {X_test.shape}")
        
        # 예측 수행
        y_pred = self.predict(X_test)
        
        # 성능 지표 계산
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 추가 통계 정보
        residuals = y_test - y_pred
        residual_std = np.std(residuals)
        
        # 결과 저장
        evaluation_results = {
            'R² Score': r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Residual Std': residual_std,
            'Mean Target': np.mean(y_test),
            'Std Target': np.std(y_test)
        }
        
        # 결과 출력
        print(f"R² 점수 (결정 계수): {r2:.4f}")
        print(f"MSE (평균 제곱 오차): {mse:.4f}")
        print(f"RMSE (평균 제곱근 오차): {rmse:.4f}")
        print(f"MAE (평균 절대 오차): {mae:.4f}")
        print(f"잔차 표준편차: {residual_std:.4f}")
        
        # 성능 해석
        self._interpret_performance(evaluation_results)
        
        return evaluation_results
    
    def _interpret_performance(self, results):
        """
        모델 성능 결과를 해석하고 출력합니다.
        
        매개변수:
        results (dict): 성능 평가 결과
        """
        print(f"\n=== 성능 해석 ===")
        
        r2 = results['R² Score']
        rmse = results['RMSE']
        mae = results['MAE']
        
        # R² 점수 해석
        if r2 >= 0.9:
            r2_interpretation = "매우 우수"
        elif r2 >= 0.7:
            r2_interpretation = "좋음"
        elif r2 >= 0.5:
            r2_interpretation = "보통"
        else:
            r2_interpretation = "개선 필요"
        
        print(f"R² 점수 {r2:.4f}는 {r2_interpretation}한 예측 성능을 나타냅니다.")
        print(f"모델이 타겟 변수 분산의 {r2*100:.1f}%를 설명합니다.")
        
        # RMSE 해석
        mean_target = results['Mean Target']
        rmse_ratio = rmse / mean_target
        print(f"RMSE {rmse:.4f}는 평균 타겟 값의 {rmse_ratio*100:.1f}%입니다.")
        
        # MAE 해석
        mae_ratio = mae / mean_target
        print(f"MAE {mae:.4f}는 평균 타겟 값의 {mae_ratio*100:.1f}%입니다.")
    
    def get_feature_importance(self):
        """
        특성 중요도를 반환합니다.
        
        선형 회귀에서 특성 중요도는 계수의 절댓값으로 정의됩니다.
        절댓값이 클수록 해당 특성이 예측에 더 큰 영향을 미칩니다.
        
        반환값:
        dict: 특성 이름과 중요도를 담은 딕셔너리
        """
        if not self.is_trained:
            raise ValueError("모델이 아직 학습되지 않았습니다.")
        
        if self.feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(len(self.model.coef_))]
        
        # 계수의 절댓값을 중요도로 사용
        importance = np.abs(self.model.coef_)
        
        # 중요도 순으로 정렬
        feature_importance = dict(zip(self.feature_names, importance))
        sorted_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        print(f"\n=== 특성 중요도 ===")
        for feature, imp in sorted_importance.items():
            print(f"{feature}: {imp:.4f}")
        
        return sorted_importance
    
    def save_model(self, filepath=None):
        """
        학습된 모델을 파일로 저장합니다.
        
        매개변수:
        filepath (str): 저장할 파일 경로 (기본값: models/ 디렉토리에 저장)
        """
        if not self.is_trained:
            raise ValueError("저장할 모델이 없습니다. 먼저 모델을 학습하세요.")
        
        if filepath is None:
            # models 디렉토리가 없으면 생성
            os.makedirs('models', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"models/{self.model_name}_{timestamp}.pkl"
        
        # 모델과 메타데이터를 함께 저장
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"모델이 '{filepath}'에 저장되었습니다.")
    
    def load_model(self, filepath):
        """
        저장된 모델을 파일에서 로딩합니다.
        
        매개변수:
        filepath (str): 로딩할 모델 파일 경로
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"모델 파일 '{filepath}'을 찾을 수 없습니다.")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        self.feature_names = model_data['feature_names']
        
        print(f"모델 '{self.model_name}'이 '{filepath}'에서 로딩되었습니다.")
        print(f"학습 상태: {'완료' if self.is_trained else '미완료'}")
    
    def get_model_info(self):
        """
        모델에 대한 정보를 반환합니다.
        
        반환값:
        dict: 모델 정보를 담은 딕셔너리
        """
        info = {
            'model_name': self.model_name,
            'model_type': 'Linear Regression',
            'is_trained': self.is_trained,
            'feature_count': len(self.model.coef_) if self.is_trained else 0
        }
        
        if self.is_trained:
            info.update({
                'intercept': self.model.intercept_,
                'coefficients': self.model.coef_.tolist(),
                'training_time': self.training_history.get('training_time', 0),
                'training_samples': self.training_history.get('training_samples', 0)
            })
        
        return info

# 사용 예시
if __name__ == "__main__":
    # 모델 인스턴스 생성
    model = DiabetesLinearRegression("test_model")
    
    # 모델 정보 출력
    info = model.get_model_info()
    print("모델 정보:", info) 