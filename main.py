"""
당뇨병 환자 선형 회귀 머신러닝 프로젝트 - 메인 실행 파일

이 파일은 당뇨병 환자 데이터를 사용한 선형 회귀 모델의
전체 학습 및 평가 파이프라인을 실행합니다.

실행 방법:
python main.py

"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trainer import DiabetesModelTrainer
from src.evaluator import DiabetesModelEvaluator

def main():
    """
    메인 실행 함수
    
    이 함수는 다음과 같은 순서로 실행됩니다:
    1. 데이터 로딩 및 전처리
    2. 선형 회귀 모델 학습
    3. 교차 검증
    4. 모델 성능 평가
    5. 결과 시각화 및 보고서 생성
    """
    print("="*70)
    print("당뇨병 환자 선형 회귀 머신러닝 프로젝트")
    print("="*70)
    print("이 프로젝트는 scikit-learn의 당뇨병 데이터셋을 사용하여")
    print("선형 회귀 모델을 학습시키고 성능을 평가합니다.")
    print("="*70)
    
    try:
        # 1. 모델 트레이너 초기화 및 전체 파이프라인 실행
        # 전체 데이터 중에서 20%를 테스트용 데이터로 분할 ,나머지 80%는 훈련(train)용
        print("\n1단계: 모델 트레이너 초기화")
        trainer = DiabetesModelTrainer(test_size=0.2, random_state=42)

        #"전체 학습 파이프라인 실행" 말의 의미: 전체 머신러닝 흐름을 순차적으로 실행하는 컨트롤러 
        # 컨트롤러가 train_model()을 내부에서 호출해서 모델(선형 회귀 모델 객체 생성하여) 학습
        print("\n2단계: 전체 학습 파이프라인 실행 (모델학습)")
        model, training_results, cv_results = trainer.run_complete_pipeline(
            model_name="diabetes_linear_regression_v1"
        )
        
        # 2. 상세 모델 평가
        print("\n3단계: 상세 모델 평가")
        evaluator = DiabetesModelEvaluator(
            model=model,
            X_test=trainer.data_loader.X_test,
            y_test=trainer.data_loader.y_test,
            feature_names=trainer.data_loader.feature_names
        )
        
        # 종합 평가 수행
        evaluation_results = evaluator.comprehensive_evaluation()
        
        # 상세 시각화 생성
        evaluator.create_detailed_visualizations()
        
        # 평가 보고서 생성
        evaluation_report = evaluator.generate_evaluation_report()
        
        # 3. 최종 결과 요약
        print("\n" + "="*70)
        print("프로젝트 실행 완료!")
        print("="*70)
        
        # 주요 결과 출력
        print("\n📊 주요 결과 요약:")
        print(f"• 모델 유형: 선형 회귀 (Linear Regression)")
        print(f"• 데이터셋: scikit-learn 당뇨병 데이터셋")
        print(f"• 훈련 샘플 수: {training_results.get('evaluation', {}).get('training_samples', 'N/A')}")
        print(f"• 테스트 샘플 수: {len(trainer.data_loader.y_test)}")
        print(f"• 특성 수: {len(trainer.data_loader.feature_names)}")
        
        # 성능 지표 출력
        if 'evaluation' in training_results:
            eval_metrics = training_results['evaluation']
            print(f"\n🎯 성능 지표:")
            print(f"• R² 점수: {eval_metrics.get('R² Score', 0):.4f}")
            print(f"• RMSE: {eval_metrics.get('RMSE', 0):.4f}")
            print(f"• MAE: {eval_metrics.get('MAE', 0):.4f}")
        
        # 교차 검증 결과 출력
        if cv_results:
            print(f"\n🔄 교차 검증 결과:")
            print(f"• 평균 R² 점수: {cv_results.get('mean_score', 0):.4f}")
            print(f"• 표준편차: {cv_results.get('std_score', 0):.4f}")
        
        # 생성된 파일들 안내
        print(f"\n📁 생성된 파일들:")
        print(f"• 데이터 분석: data/diabetes_analysis.png")
        print(f"• 학습 결과: data/learning_results.png")
        print(f"• 상세 평가: data/detailed_evaluation.png")
        print(f"• 학습된 모델: models/ 디렉토리")
        
        print(f"\n✅ 프로젝트가 성공적으로 완료되었습니다!")
        print(f"생성된 시각화 파일들을 확인하여 결과를 분석하세요.")
        
    except Exception as e:
        print(f"\n❌ 오류가 발생했습니다: {str(e)}")
        print("오류를 확인하고 다시 실행해주세요.")
        return False
    
    return True

def run_quick_test():
    """
    빠른 테스트 실행 함수
    
    이 함수는 기본적인 모델 학습과 평가만을 수행하여
    프로젝트가 정상적으로 작동하는지 확인합니다.
    """
    print("="*50)
    print("빠른 테스트 실행")
    print("="*50)
    
    try:
        # 트레이너 초기화
        trainer = DiabetesModelTrainer(test_size=0.3, random_state=42)
        
        # 데이터 설정 (시각화 제외)
        X_train, X_test, y_train, y_test, feature_names = trainer.setup_data()
        
        # 모델 학습
        model = trainer.train_model("quick_test_model", feature_names, save_model=False)
        
        # 기본 평가
        evaluation_results = model.evaluate(X_test, y_test)
        
        print(f"\n✅ 빠른 테스트 완료!")
        print(f"R² 점수: {evaluation_results.get('R² Score', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 빠른 테스트 실패: {str(e)}")
        return False

if __name__ == "__main__":
    # 명령행 인수 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # 빠른 테스트 모드
        success = run_quick_test()
    else:
        # 전체 파이프라인 실행
        success = main()
    
    # 종료 코드 설정
    sys.exit(0 if success else 1) 