# 💸은행 데이터 모델링 프로젝트💰
pycaret 라이브러리를 활용하여 은행 데이터를 분석하고, 최적의 머신러닝 모델을 찾습니다. <br>
이 과정에서 데이터 전처리, 모델 비교, 그리고 모델 튜닝을 포함한 전반적인 분석을 수행합니다.

## 목차
1. 프로젝트 소개<br>
2. 기술 스택<br>
3. 설치<br>
4. 데이터셋<br>
5. 모델링 과정<br>
6. 모델 튜닝<br>
7. 결과<br>
8. 참고<br>
9. 프로젝트 소개<br>

## 기술 스택
  ✔ Python<br>
  ✔ Pandas<br>
  ✔ Seaborn<br>
  ✔ Matplotlib<br>
  ✔ Scikit-learn<br>
  ✔ PyCaret<br>
  ✔ Optuna

## 설치
```
!pip install pycaret
!pip install seaborn scikit-learn
!pip install optuna
!pip install pycaret[tuners]
```

## 데이터셋
```
# 데이터 로드
df_bank = pd.read_csv('bank.csv')
```

## 모델링 과정
⑴ PyCaret 설정
```
from pycaret.classification import ClassificationExperiment

s = ClassificationExperiment()
s.setup(df_bank, target='deposit', session_id=123)
```

⑵ 모델 비교<br>
PyCaret의 compare_models 기능을 통해 여러 모델을 비교하여 최적의 모델 찾기
```
from pycaret.classification import compare_models

best = s.compare_models()
print(best)
```
![image (1)](https://github.com/user-attachments/assets/161c94d2-dfbf-41a8-aada-ce9016ee2ce3)

![image (2)](https://github.com/user-attachments/assets/9ba6106d-48cb-4aed-b253-d66ddd6eedf9)

## 모델 튜닝
최적의 모델을 선택한 후, optuna 라이브러리를 활용하여 모델 튜닝
```
best_model = s.tune_model(best, optimize='F1', search_library='optuna')
print(best_model)
```

## 결과
모델을 비교하고 튜닝하여 최적의 모델을 도출함.<br>
F1 스코어를 최적화하여 모델 성능을 향상시킴

![image](https://github.com/user-attachments/assets/b3a5c839-96e1-4b0d-8c03-6674d7c9cf29)

---

## 트러블슈팅 : 시각화 문제 해결
PyCaret의 ClassificationExperiment 객체를 사용하여 설정할 때 시각화에 문제가 발생<br><br>
  ① setup() 호출 후 환경 확인
  setup() 메서드가 정상적으로 완료된 후, PyCaret 환경이 올바르게 설정되었는지 확인해야 함. 이를 확인한 후에 plot_model()을 호출.
  
  ② 모델 시각화
  setup() 메서드가 정상적으로 완료되면, 선택된 최적의 모델에 대해 피처 중요도를 시각화할 수 있음.
```
# PyCaret 환경 설정 후 피처 중요도 시각화
s.setup(df_bank, target='deposit', session_id=123)

# 모델 시각화
best_model = s.compare_models()
s.plot_model(best_model, plot='feature')
```
