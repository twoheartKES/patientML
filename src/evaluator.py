"""
모델 평가 및 분석 모듈

이 모듈은 학습된 선형 회귀 모델의 성능을 심층적으로 평가하고,
모델의 예측 능력을 분석하는 기능을 제공합니다.

주요 기능:
- 모델 성능 심층 분석
- 예측 오차 분석
- 모델 해석성 분석
- 성능 비교 및 벤치마킹
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DiabetesModelEvaluator:
    """
    당뇨병 선형 회귀 모델을 심층적으로 평가하는 클래스
    
    이 클래스는 학습된 모델의 성능을 다양한 관점에서 분석하고,
    모델의 예측 능력과 신뢰성을 평가합니다.
    """
    
    def __init__(self, model, X_test, y_test, feature_names=None):
        """
        모델 평가자 초기화
        
        매개변수:
        model: 학습된 선형 회귀 모델
        X_test (array-like): 테스트 특성 데이터
        y_test (array-like): 테스트 타겟 데이터
        feature_names (list): 특성 이름들의 리스트
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.y_pred = None
        self.residuals = None
        self.evaluation_results = {}
        
        # 예측 수행
        if hasattr(model, 'predict'):
            self.y_pred = model.predict(X_test)
            self.residuals = y_test - self.y_pred
        
        print("모델 평가자가 초기화되었습니다.")
    
    def comprehensive_evaluation(self):
        """
        모델의 종합적인 성능을 평가합니다.
        
        이 메서드는 다양한 성능 지표와 통계적 분석을 통해
        모델의 예측 능력을 종합적으로 평가합니다.
        
        반환값:
        dict: 종합 평가 결과
        """
        print("=== 종합 모델 평가 시작 ===")
        
        if self.y_pred is None:
            raise ValueError("예측 결과가 없습니다. 모델을 확인하세요.")
        
        # 기본 성능 지표 계산
        basic_metrics = self._calculate_basic_metrics()
        
        # 통계적 분석
        statistical_analysis = self._statistical_analysis()
        
        # 오차 분석
        error_analysis = self._error_analysis()
        
        # 모델 해석성 분석
        interpretability_analysis = self._interpretability_analysis()
        
        # 종합 평가 결과
        self.evaluation_results = {
            '기본 성능 지표': basic_metrics,
            '통계적 분석': statistical_analysis,
            '오차 분석': error_analysis,
            '해석성 분석': interpretability_analysis
        }
        
        # 결과 출력
        self._print_comprehensive_results()
        
        return self.evaluation_results
    
    def _calculate_basic_metrics(self):
        """
        기본 성능 지표를 계산합니다.
        
        반환값:
        dict: 기본 성능 지표
        """
        # 기본 지표 계산
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        # 추가 지표
        mape = np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100
        explained_variance = 1 - np.var(self.residuals) / np.var(self.y_test)
        
        return {
            'R² Score': r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE (%)': mape,
            'Explained Variance': explained_variance
        }
    
    def _statistical_analysis(self):
        """
        통계적 분석을 수행합니다.
        
        반환값:
        dict: 통계적 분석 결과
        """
        # 잔차 통계
        residual_mean = np.mean(self.residuals)
        residual_std = np.std(self.residuals)
        residual_skew = stats.skew(self.residuals)
        residual_kurtosis = stats.kurtosis(self.residuals)
        
        # 정규성 검정 (Shapiro-Wilk test)
        shapiro_stat, shapiro_p = stats.shapiro(self.residuals)
        
        # 잔차의 자기상관 검정 (Durbin-Watson test)
        dw_stat = self._durbin_watson_test()
        
        return {
            '잔차 평균': residual_mean,
            '잔차 표준편차': residual_std,
            '잔차 왜도': residual_skew,
            '잔차 첨도': residual_kurtosis,
            '정규성 검정 (Shapiro-Wilk)': {
                '통계량': shapiro_stat,
                'p-value': shapiro_p,
                '정규성': '정규분포' if shapiro_p > 0.05 else '비정규분포'
            },
            '자기상관 검정 (Durbin-Watson)': {
                '통계량': dw_stat,
                '해석': self._interpret_dw_statistic(dw_stat)
            }
        }
    
    def _error_analysis(self):
        """
        예측 오차를 분석합니다.
        
        반환값:
        dict: 오차 분석 결과
        """
        # 오차 분포 분석
        error_percentiles = np.percentile(self.residuals, [5, 10, 25, 50, 75, 90, 95])
        
        # 이상치 탐지 (IQR 방법)
        q1, q3 = np.percentile(self.residuals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((self.residuals < lower_bound) | (self.residuals > upper_bound))
        
        # 오차 패턴 분석
        error_pattern = self._analyze_error_patterns()
        
        return {
            '오차 분위수': {
                '5%': error_percentiles[0],
                '10%': error_percentiles[1],
                '25%': error_percentiles[2],
                '50%': error_percentiles[3],
                '75%': error_percentiles[4],
                '90%': error_percentiles[5],
                '95%': error_percentiles[6]
            },
            '이상치 분석': {
                '이상치 개수': outliers,
                '이상치 비율 (%)': (outliers / len(self.residuals)) * 100,
                '하한': lower_bound,
                '상한': upper_bound
            },
            '오차 패턴': error_pattern
        }
    
    def _interpretability_analysis(self):
        """
        모델의 해석성을 분석합니다.
        
        반환값:
        dict: 해석성 분석 결과
        """
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'coef_'):
            return {'error': '모델 계수 정보를 찾을 수 없습니다.'}
        
        coefficients = self.model.model.coef_
        intercept = self.model.model.intercept_
        
        # 계수 통계
        coef_mean = np.mean(np.abs(coefficients))
        coef_std = np.std(coefficients)
        coef_range = np.max(coefficients) - np.min(coefficients)
        
        # 특성별 영향력 분석
        feature_impact = {}
        if self.feature_names:
            for i, feature in enumerate(self.feature_names):
                feature_impact[feature] = {
                    '계수': coefficients[i],
                    '절댓값': abs(coefficients[i]),
                    '영향력 순위': np.argsort(np.abs(coefficients))[::-1][i] + 1
                }
        
        return {
            '절편': intercept,
            '계수 통계': {
                '평균 절댓값': coef_mean,
                '표준편차': coef_std,
                '범위': coef_range
            },
            '특성별 영향력': feature_impact
        }
    
    def _durbin_watson_test(self):
        """
        Durbin-Watson 통계량을 계산합니다.
        
        반환값:
        float: Durbin-Watson 통계량
        """
        diff_residuals = np.diff(self.residuals)
        dw_stat = np.sum(diff_residuals**2) / np.sum(self.residuals**2)
        return dw_stat
    
    def _interpret_dw_statistic(self, dw_stat):
        """
        Durbin-Watson 통계량을 해석합니다.
        
        매개변수:
        dw_stat (float): Durbin-Watson 통계량
        
        반환값:
        str: 해석 결과
        """
        if dw_stat < 1.5:
            return "양의 자기상관 (잔차가 연속적으로 같은 방향)"
        elif dw_stat > 2.5:
            return "음의 자기상관 (잔차가 연속적으로 반대 방향)"
        else:
            return "자기상관 없음 (잔차가 독립적)"
    
    def _analyze_error_patterns(self):
        """
        오차 패턴을 분석합니다.
        
        반환값:
        dict: 오차 패턴 분석 결과
        """
        # 예측값에 따른 오차 패턴
        pred_residual_corr = np.corrcoef(self.y_pred, self.residuals)[0, 1]
        
        # 실제값에 따른 오차 패턴
        actual_residual_corr = np.corrcoef(self.y_test, self.residuals)[0, 1]
        
        # 오차의 분산 패턴 (heteroscedasticity 검정)
        # 예측값을 구간별로 나누어 분산 비교
        n_bins = 5
        bin_edges = np.linspace(self.y_pred.min(), self.y_pred.max(), n_bins + 1)
        bin_variances = []
        
        for i in range(n_bins):
            mask = (self.y_pred >= bin_edges[i]) & (self.y_pred < bin_edges[i + 1])
            if np.sum(mask) > 1:
                bin_variances.append(np.var(self.residuals[mask]))
        
        variance_ratio = max(bin_variances) / min(bin_variances) if len(bin_variances) > 1 else 1
        
        return {
            '예측값-잔차 상관관계': pred_residual_corr,
            '실제값-잔차 상관관계': actual_residual_corr,
            '이분산성 검정': {
                '분산 비율': variance_ratio,
                '해석': '이분산성 있음' if variance_ratio > 4 else '등분산성'
            }
        }
    
    def _print_comprehensive_results(self):
        """
        종합 평가 결과를 출력합니다.
        """
        print("\n" + "="*60)
        print("종합 모델 평가 결과")
        print("="*60)
        
        for section, content in self.evaluation_results.items():
            print(f"\n[{section}]")
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            print(f"    {sub_key}: {sub_value}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {content}")
    
    def create_detailed_visualizations(self):
        """
        상세한 시각화를 생성합니다.
        """
        print("=== 상세 시각화 생성 중 ===")
        
        # 그래프 스타일 설정
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('당뇨병 선형 회귀 모델 상세 평가', fontsize=16, fontweight='bold')
        
        # 1. 실제값 vs 예측값 (메인 플롯)
        axes[0, 0].scatter(self.y_test, self.y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('실제값')
        axes[0, 0].set_ylabel('예측값')
        axes[0, 0].set_title('실제값 vs 예측값')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 플롯
        axes[0, 1].scatter(self.y_pred, self.residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('예측값')
        axes[0, 1].set_ylabel('잔차')
        axes[0, 1].set_title('잔차 플롯')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 잔차 히스토그램
        axes[0, 2].hist(self.residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 2].set_xlabel('잔차')
        axes[0, 2].set_ylabel('빈도')
        axes[0, 2].set_title('잔차 분포')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Q-Q 플롯 (정규성 검정)
        stats.probplot(self.residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q 플롯 (정규성 검정)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 잔차 vs 실제값
        axes[1, 1].scatter(self.y_test, self.residuals, alpha=0.6, color='orange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('실제값')
        axes[1, 1].set_ylabel('잔차')
        axes[1, 1].set_title('잔차 vs 실제값')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 예측 오차 분포
        error_percentages = np.abs(self.residuals / self.y_test) * 100
        axes[1, 2].hist(error_percentages, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 2].set_xlabel('예측 오차 (%)')
        axes[1, 2].set_ylabel('빈도')
        axes[1, 2].set_title('예측 오차 분포')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. 특성별 계수 (있는 경우)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'coef_'):
            coefficients = self.model.model.coef_
            if self.feature_names:
                y_pos = np.arange(len(self.feature_names))
                colors = ['red' if coef < 0 else 'blue' for coef in coefficients]
                axes[2, 0].barh(y_pos, coefficients, color=colors, alpha=0.7)
                axes[2, 0].set_yticks(y_pos)
                axes[2, 0].set_yticklabels(self.feature_names)
                axes[2, 0].set_xlabel('계수 값')
                axes[2, 0].set_title('특성별 계수')
                axes[2, 0].grid(True, alpha=0.3)
            else:
                axes[2, 0].bar(range(len(coefficients)), coefficients, alpha=0.7)
                axes[2, 0].set_xlabel('특성 인덱스')
                axes[2, 0].set_ylabel('계수 값')
                axes[2, 0].set_title('특성별 계수')
                axes[2, 0].grid(True, alpha=0.3)
        
        # 8. 성능 지표 요약
        if self.evaluation_results:
            basic_metrics = self.evaluation_results.get('기본 성능 지표', {})
            metrics = ['R² Score', 'RMSE', 'MAE']
            values = [basic_metrics.get('R² Score', 0), 
                     basic_metrics.get('RMSE', 0), 
                     basic_metrics.get('MAE', 0)]
            
            bars = axes[2, 1].bar(metrics, values, color=['green', 'red', 'orange'])
            axes[2, 1].set_ylabel('값')
            axes[2, 1].set_title('성능 지표 요약')
            axes[2, 1].grid(True, alpha=0.3)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.4f}', ha='center', va='bottom')
        
        # 9. 잔차 자기상관 플롯
        axes[2, 2].scatter(self.residuals[:-1], self.residuals[1:], alpha=0.6, color='purple')
        axes[2, 2].set_xlabel('잔차 (t)')
        axes[2, 2].set_ylabel('잔차 (t+1)')
        axes[2, 2].set_title('잔차 자기상관')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/detailed_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("상세 시각화 결과가 'data/detailed_evaluation.png'에 저장되었습니다.")
    
    def generate_evaluation_report(self):
        """
        평가 결과에 대한 상세한 보고서를 생성합니다.
        
        반환값:
        dict: 평가 보고서
        """
        if not self.evaluation_results:
            print("먼저 comprehensive_evaluation()을 실행하세요.")
            return None
        
        print("=== 평가 보고서 생성 중 ===")
        
        # 보고서 구성
        report = {
            '평가 개요': {
                '평가 일시': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                '테스트 샘플 수': len(self.y_test),
                '특성 수': self.X_test.shape[1] if self.X_test is not None else 'N/A'
            },
            '평가 결과': self.evaluation_results
        }
        
        # 모델 권장사항 추가
        recommendations = self._generate_recommendations()
        report['모델 권장사항'] = recommendations
        
        return report
    
    def _generate_recommendations(self):
        """
        평가 결과를 바탕으로 모델 개선 권장사항을 생성합니다.
        
        반환값:
        dict: 권장사항
        """
        recommendations = []
        
        # R² 점수 기반 권장사항
        r2_score = self.evaluation_results.get('기본 성능 지표', {}).get('R² Score', 0)
        if r2_score < 0.5:
            recommendations.append("모델 성능이 낮습니다. 특성 엔지니어링이나 다른 알고리즘을 고려하세요.")
        elif r2_score < 0.7:
            recommendations.append("모델 성능이 보통입니다. 하이퍼파라미터 튜닝을 고려하세요.")
        
        # 정규성 검정 기반 권장사항
        normality_test = self.evaluation_results.get('통계적 분석', {}).get('정규성 검정 (Shapiro-Wilk)', {})
        if normality_test.get('정규성') == '비정규분포':
            recommendations.append("잔차가 정규분포를 따르지 않습니다. 데이터 변환이나 다른 모델을 고려하세요.")
        
        # 자기상관 검정 기반 권장사항
        dw_interpretation = self.evaluation_results.get('통계적 분석', {}).get('자기상관 검정 (Durbin-Watson)', {}).get('해석', '')
        if '자기상관' in dw_interpretation:
            recommendations.append("잔차에 자기상관이 있습니다. 시계열 모델이나 다른 접근법을 고려하세요.")
        
        # 이분산성 검정 기반 권장사항
        heteroscedasticity = self.evaluation_results.get('오차 분석', {}).get('오차 패턴', {}).get('이분산성 검정', {}).get('해석', '')
        if '이분산성' in heteroscedasticity:
            recommendations.append("이분산성이 발견되었습니다. 가중 최소제곱법이나 데이터 변환을 고려하세요.")
        
        return recommendations

# 사용 예시
if __name__ == "__main__":
    print("DiabetesModelEvaluator 모듈이 로드되었습니다.")
    print("사용법: evaluator = DiabetesModelEvaluator(model, X_test, y_test)") 