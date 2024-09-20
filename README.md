# 💸은행 데이터 모델링 프로젝트💰
pycaret 라이브러리를 활용하여 은행 데이터를 분석하고, 예금 여부를 분류 및 예측하는 최적의 머신러닝 모델을 찾습니다. <br>
이 과정에서 데이터 전처리, 모델 비교, 그리고 모델 튜닝을 포함한 전반적인 분석을 수행합니다.

## 🚌 목차
1. 프로젝트 소개<br>
2. 기술 스택<br>
3. 설치<br>
4. 데이터셋<br>
5. 모델링 과정<br>
6. 모델 튜닝<br>
7. 결과<br>
8. 트러블 슈팅 및 추가할 점<br>

<br>

##  🖐 팀원소개
|    소속    |   과정   |   이름  |
| :--------: |  :----: |   :----: |
| 우리_fis 아카데미 3기 | AI 엔지니어링 | 배희진 |
| 우리_fis 아카데미 3기 | AI 엔지니어링 | 서정윤 |
| 우리_fis 아카데미 3기 | AI 엔지니어링 | 이훈표 |
| 우리_fis 아카데미 3기 | AI 엔지니어링 | 함동미 |

<br>

## 🎀 데이터셋 출처
https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset?select=bank.csv

<br>

## 🎪 기술 스택

  ✓ Python<br>
  ✓ Pandas<br>
  ✓ Seaborn<br>
  ✓ Matplotlib<br>
  ✓ Scikit-learn<br>
  ✓ AutoML(PyCaret)<br>
  ✓ Optuna

<br>

## 🎢 설치
```
!pip install pycaret
!pip install seaborn scikit-learn
!pip install optuna
!pip install pycaret[tuners]
```

## 🛒 데이터셋
```
# 데이터 로드
df_bank = pd.read_csv('bank.csv')
```

## 🎠 모델링 과정
⑴ PyCaret 설정
```
from pycaret.classification import ClassificationExperiment

# PyCaret 설정: 전진 선택법과 함께 피처 선택
s = ClassificationExperiment()

# setup의 결과를 'exp' 변수에 저장
exp = s.setup(df_bank,
              target='deposit',
              session_id=123,
              feature_selection=False)  # 전진 선택법 비활성화
```

⑵ 모델 비교<br>
PyCaret의 compare_models 기능을 통해 여러 모델을 비교하여 최적의 모델 찾기
```
from pycaret.classification import compare_models

best_model = s.compare_models()  # 모델 비교 후 가장 좋은 모델 선택
```

![bestmodel](https://github.com/user-attachments/assets/ed306cd3-e7e1-431d-90a3-da2e5c39d385)

![printbestmodel](https://github.com/user-attachments/assets/ce1087aa-5b95-4ca9-aacf-ac43192868d6)


## 🎫 모델 튜닝
최적의 모델을 선택한 후, optuna 라이브러리를 활용하여 모델 튜닝
```
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 원본 데이터프레임에서 'deposit' 컬럼 분리
X = df_bank.drop('deposit', axis=1)
y = df_bank['deposit']

# 범주형 변수 인코딩
X_encoded = pd.get_dummies(X, drop_first=True)  # one-hot 인코딩

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=123)

# Objective function for Optuna
def objective(trial):
    param = {
        'objective': 'binary', # 목표 함수 설정: 이진 분류 문제
        'metric': 'binary_error',  # 평가 지표: 이진 오류
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
    }

    model = lgb.LGBMClassifier(**param) # 모델초기화
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    return accuracy

# Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Best parameters
print(study.best_params)
```

## 🎉 결과
모델을 비교하고 튜닝하여 최적의 모델을 도출함.<br>
F1 스코어를 최적화하여 모델 성능을 향상시킴

![최종결과](https://github.com/user-attachments/assets/787ad9a5-773b-425c-9cc2-e466d82ea448)

![최종결과2](https://github.com/user-attachments/assets/6a75bd43-fc9f-43aa-b4cb-29b7c469dda2)

![최종결과3](https://github.com/user-attachments/assets/c1408d6e-a449-4f8d-896a-cdde22391442)

---

## 🎇 트러블슈팅 : 시각화 문제 해결
1. PyCaret의 ClassificationExperiment 객체를 사용하여 설정할 때 시각화에 문제가 발생<br>
    파이캐럿에서 setup() 함수를 사용할 때,  s.setup()이 적절한 컨텍스트를 제공하지 않아서 시각화가 이루어지지 않음. <br>
    제대로 작동하려면 필요한 정보나 상태가 미리 설정되어 있어야하는데 이 설정이 특정 객체에 바인딩되지 않으면 시각화가 제대로 작동하지않음.
    따라서, setup의 결과를 exp 변수에 저장해주었음

    ```
    <전>
    s = ClassificationExperiment()
    s.setup(df_bank, target = 'deposit', session_id = 123)

    <후>
    # PyCaret 환경 설정 후 피처 중요도 시각화
    exp = s.setup(df_bank,
                  target='deposit',
                  session_id=123,
                  feature_selection=False) 
    ```

2. 처음에 Optuna를 tune_model함수의 search_library에서 옵션으로 추가해 돌려줌. <br>
   Optuna를 안 만들고 옵션 안에 넣어 시각화 불가능 -> Optuna를 옵션 안에 넣지 않고 따로 설정해주었음.
   
    ```
    <전>
    best_model = s.tune_model(best, optimize='F1', search_library='optuna')

    <후>
    # Objective function for Optuna
    def objective(trial):
        param = {
            'objective': 'binary', # 목표 함수 설정: 이진 분류 문제
            'metric': 'binary_error',  # 평가 지표: 이진 오류
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
        }
    
        model = lgb.LGBMClassifier(**param) # 모델초기화
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
    
        return accuracy
    ```
    
  3. Duration은 사후에 기록된 컬럼이기 때문에, 앞으로의 예측에서는 수집할 수 없는 데이터이므로 drop해줘야함. <br>
     해당 피처를 제거해주지 않아 정확한 결과 도출이 어려움 -> 제거 후 분석 진행

      ```
    <전>
  

    <후>
    df_bank = df_bank.drop('duration', axis=1)
    ```
      
---

## 🎁 추가할 점 
① 파라미터에 따른 결과를 비교해 볼 수 있도록 Streamlit으로 배포까지 진행하고싶다. <br>
② Optuna Dashboard를 사용해보고 싶다.

---

## 👀 피드백
Optuna Dashboard를 사용해보자!

