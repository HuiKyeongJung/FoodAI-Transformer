# 🍽️ FoodAI - 저메추 AI

> Transformer의 **Text Classification(Task)** 를 활용하여 사용자의 상황을 분석하고, 가장 적합한 음식을 **Top3**로 추천하는 AI 서비스입니다.

---

# 📌 Project Overview

FoodAI는 사용자가 입력한 문장을 자연어 처리(NLP)를 통해 분석하여 현재 상황에 가장 적합한 음식을 추천하는 프로젝트입니다.

본 프로젝트는 Hugging Face의 **KLUE-RoBERTa-small** 모델을 Fine-tuning하여 음식을 분류하고, 예측 확률이 높은 Top3 음식을 추천하도록 구현하였습니다.

---

# 🎯 Features

- 사용자의 상황 문장 입력
- Transformer 기반 자연어 처리
- 음식 11종 분류
- Top3 음식 추천
- 추천 확률(Confidence Score) 출력
- Epoch 하이퍼파라미터 실험 및 성능 비교

---

# 🤖 Model

### Model

- **KLUE-RoBERTa-small**

### NLP Task

- **Text Classification**
- 사용자의 문장을 분석하여 11개의 음식 클래스 중 가장 적합한 음식을 분류하는 다중 클래스 분류(Multi-class Classification) Task

### Transformer Structure

- **Encoder-only Transformer**

본 프로젝트는 문장을 생성하는 모델이 아니라 입력 문장을 분류하는 모델이므로 Decoder를 사용하지 않습니다.

---

# 🔍 Attention Mechanism

### Bidirectional Multi-Head Self-Attention

본 프로젝트는 **Self-Attention**을 사용하여 입력 문장의 각 단어가 서로를 참고하면서 문맥을 이해합니다.

또한 여러 개의 Attention Head가 서로 다른 관점에서 문장 정보를 학습하는 **Multi-Head Self-Attention** 구조를 사용하며, 양방향(Bidirectional) 문맥을 모두 고려하여 문장의 의미를 효과적으로 파악합니다.

> ❌ Cross-Attention은 Encoder와 Decoder를 함께 사용하는 생성(Task)에서 사용되므로 본 프로젝트에서는 사용되지 않습니다.

---

# 🍽 Food Labels

총 **11개의 음식 클래스**

- 김밥
- 떡볶이
- 마라탕
- 빙수
- 삼계탕
- 소고기
- 죽
- 치킨
- 칼국수
- 파전
- 피자

---

# ⚙ Hyperparameter

| Parameter | Value |
|-----------|------|
| Model | KLUE-RoBERTa-small |
| Max Length | 64 |
| Learning Rate | 2e-5 |
| Batch Size | 8 |
| Loss Function | Cross Entropy Loss |

---

# 📊 Hyperparameter Experiment

## num_train_epochs = 3

| Metric | Result |
|--------|--------|
| Accuracy | 59.09% |
| F1-score | 56.37% |

---

## num_train_epochs = 5

| Metric | Result |
|--------|--------|
| Accuracy | 67.27% |
| F1-score | 66.16% |

---

## num_train_epochs = 7

| Metric | Result |
|--------|--------|
| Accuracy | **70.00%** |
| F1-score | **69.80%** |

---

# 📈 Result

| num_train_epochs | Accuracy | F1-score |
|-----------------|---------:|---------:|
| 3 | 59.09% | 56.37% |
| 5 | 67.27% | 66.16% |
| **7** | **70.00%** | **69.80%** |

### Conclusion

하이퍼파라미터인 **num_train_epochs**를 3, 5, 7로 변경하여 성능을 비교한 결과,

**num_train_epochs = 7**

에서 가장 높은 Accuracy(70.00%)와 F1-score(69.80%)를 기록하였습니다.

---

# 💻 How to Run

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

### Predict

```bash
python predict.py
```

---

# 🖥 Example

```
상황을 입력하세요.

오늘 너무 덥다.

🍽 추천 음식 TOP3

🥇 빙수
🥈 칼국수
🥉 삼계탕
```

---

# 📁 Project Structure

```
FoodAI-Transformer
│
├── train.py
├── predict.py
├── data.csv
├── requirements.txt
├── label_encoder.pkl
├── food_model/
└── README.md
```

---

# 📚 What I Learned

- Transformer 기반 Text Classification 구현
- KLUE-RoBERTa-small Fine-tuning
- Self-Attention 메커니즘 이해
- 하이퍼파라미터(Epoch) 실험 및 성능 분석
- Accuracy와 F1-score를 활용한 모델 평가
- Hugging Face Transformers 활용
- Git & GitHub를 통한 프로젝트 관리

---

# 👨‍💻 Author

**정희경**

Soonchunhyang University

Medical IT Engineering