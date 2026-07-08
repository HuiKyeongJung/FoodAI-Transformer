# 🍽️ FoodAI - 저메추 AI

> Transformer 기반 **Text Classification** Task를 활용하여 사용자의 상황을 분석하고, 가장 적합한 음식을 Top3로 추천하는 AI 서비스입니다.

---

## 📌 프로젝트 소개

FoodAI는 사용자가 입력한 문장을 Transformer 모델이 분석하여 상황에 가장 어울리는 음식을 추천하는 자연어 처리(NLP) 프로젝트입니다.

KLUE-RoBERTa-small 모델을 Fine-tuning하여 음식을 분류하고, 각 음식에 대한 확률을 계산하여 Top3 추천 결과를 제공합니다.

---

## 🎯 주요 기능

- 사용자의 상황 문장 입력
- Transformer 기반 텍스트 분류(Text Classification)
- 음식 추천 Top3 출력
- 추천 확률(Confidence) 제공
- 하이퍼파라미터(Epoch) 실험 및 성능 비교

---

## 🛠 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- KLUE-RoBERTa-small
- Scikit-learn
- Pandas

---

## 📂 음식 라벨

총 **11개 음식 클래스**

```
김밥
떡볶이
마라탕
빙수
삼계탕
소고기
죽
치킨
칼국수
파전
피자
```

---

## ⚙️ Model

- Model : KLUE-RoBERTa-small
- Task : Text Classification
- Max Length : 64
- Optimizer : AdamW (Transformers Trainer 기본값)
- Loss : Cross Entropy Loss

---

## 📊 Hyperparameter Experiment

### num_train_epochs = 3

| Metric | Result |
|--------|--------|
| Accuracy | 59.09% |
| F1-score | 56.37% |

---

### num_train_epochs = 5

| Metric | Result |
|--------|--------|
| Accuracy | 67.27% |
| F1-score | 66.16% |

---

### num_train_epochs = 7

| Metric | Result |
|--------|--------|
| Accuracy | **70.00%** |
| F1-score | **69.80%** |

---

## 📈 Experiment Result

하이퍼파라미터인 `num_train_epochs`를 3, 5, 7로 변경하며 모델을 학습한 결과,

**num_train_epochs = 7**

에서 가장 높은 성능을 확인하였다.

| Hyperparameter | Accuracy | F1-score |
|---------------|---------:|---------:|
| Epoch = 3 | 59.09% | 56.37% |
| Epoch = 5 | 67.27% | 66.16% |
| **Epoch = 7** | **70.00%** | **69.80%** |

---

## 💻 실행 방법

### 1. 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 학습

```bash
python train.py
```

### 3. 음식 추천

```bash
python predict.py
```

---

## 🖥 실행 예시

```
상황을 입력하세요 :

오늘 너무 덥다...

🍽️ 추천 음식 TOP3

🥇 빙수
🥈 칼국수
🥉 삼계탕
```

---

## 📁 프로젝트 구조

```
FoodAI-Transformer
│
├── train.py
├── predict.py
├── data.csv
├── requirements.txt
├── label_encoder.pkl
├── food_model
└── README.md
```

---

## 📚 What I Learned

- Transformer 기반 Text Classification 구현
- KLUE-RoBERTa-small Fine-tuning
- Hugging Face Trainer 활용
- 하이퍼파라미터(Epoch) 실험 및 성능 비교
- Accuracy와 F1-score를 활용한 모델 평가
- Git & GitHub를 활용한 프로젝트 관리

---

## 👨‍💻 Author

**정희경**

Soonchunhyang University  
Medical IT Engineering