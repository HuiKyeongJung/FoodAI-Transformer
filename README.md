# 🍽️ FoodAI - 저메추 AI

> Transformer의 **Text Classification(Task)** 를 활용하여 사용자의 상황을 분석하고, 가장 적합한 음식을 **Top3**로 추천하는 AI 서비스입니다.

---

# 📌 Project Overview

FoodAI는 사용자가 입력한 문장을 자연어 처리(NLP)를 통해 분석하여 현재 상황에 가장 적합한 음식을 추천하는 프로젝트입니다.

Hugging Face의 **KLUE-RoBERTa-small** 모델을 Fine-tuning하여 11개의 음식 클래스로 분류하고, 가장 높은 확률의 음식 Top3를 추천하도록 구현하였습니다.

---

## 📊 실험 결과 페이지

FoodAI의 하이퍼파라미터 실험 결과는 별도 HTML 페이지로 정리했습니다.

👉 [FoodAI 하이퍼파라미터 실험 결과 보기](https://huikyeongjung.github.io/FoodAI-Transformer/foodai_hyperparameter_report.html)

# 📋 Project Information

| Category | Description |
|-----------|-------------|
| **Project** | FoodAI - 저메추 AI |
| **Task** | Text Classification |
| **Transformer** | Encoder-only |
| **Model** | KLUE-RoBERTa-small |
| **Attention** | Bidirectional Multi-Head Self-Attention |
| **Output** | 음식 Top3 추천 |

---

# 🔄 Model Pipeline

```text
사용자 입력 문장
        │
        ▼
   Tokenizer
        │
        ▼
KLUE-RoBERTa Encoder
(Bidirectional Multi-Head Self-Attention)
        │
        ▼
 Text Classification
        │
        ▼
11개 음식 클래스 확률 계산
        │
        ▼
 🍽️ Top3 음식 추천
```

---

# 🤖 Model

### ✔ NLP Task

**Text Classification**

사용자가 입력한 문장을 분석하여 **11개의 음식 클래스 중 하나로 분류하는 다중 클래스 분류(Multi-class Classification)** Task를 수행합니다.

---

### ✔ Transformer Structure

**Encoder-only Transformer**

본 프로젝트는 문장을 생성하는 모델이 아닌 **문장 분류(Task)** 를 수행하기 때문에 **Encoder만 사용하는 KLUE-RoBERTa** 모델을 사용하였습니다.

---

### ✔ Attention Mechanism

**Bidirectional Multi-Head Self-Attention**

- 입력 문장의 각 단어가 서로를 참고(Self-Attention)하여 문맥을 이해합니다.
- 여러 개의 Attention Head가 서로 다른 관점에서 문장을 동시에 학습하여 의미를 더욱 풍부하게 표현합니다.

> **Cross-Attention은 Encoder와 Decoder를 함께 사용하는 생성 모델에서 사용되므로 본 프로젝트에서는 사용하지 않았습니다.**

---

# 🍽 Food Labels

총 **11개의 음식 클래스**

| 음식 |
|------|
| 김밥 |
| 떡볶이 |
| 마라탕 |
| 빙수 |
| 삼계탕 |
| 소고기 |
| 죽 |
| 치킨 |
| 칼국수 |
| 파전 |
| 피자 |

---

# ⚙ Hyperparameters

| Parameter | Value |
|-----------|------:|
| Model | KLUE-RoBERTa-small |
| Learning Rate | 2e-5 |
| Batch Size | 8 |
| Max Length | 64 |
| num_train_epochs | 3 / 5 / 7 |

---

# 📊 Hyperparameter Experiment

## Performance Comparison

| num_train_epochs | Best Epoch | Accuracy | F1-score |
|:----------------:|:----------:|:--------:|:--------:|
| 3 | Epoch 2 (Accuracy) / Epoch 3 (F1) | 59.09% | 56.37% |
| 5 | Epoch 4 | 67.27% | 66.29% |
| **7** | **Epoch 7** | **70.00%** | **69.80%** |

---

## Accuracy Trend

| Epoch Setting | Accuracy |
|--------------:|---------:|
| 3 | ████████████ 59.09% |
| 5 | ████████████████ 67.27% |
| **7** | █████████████████ 70.00% |

---

## num_train_epochs = 3

| Epoch | Accuracy | F1-score |
|------:|---------:|---------:|
|1|29.09%|20.68%|
|2|60.00%|56.11%|
|3|59.09%|56.37%|

### Analysis

- Epoch 1 → 2에서 성능이 크게 향상됨.
- Accuracy는 Epoch 2에서 가장 높았으며, F1-score는 Epoch 3에서 가장 높게 나타남.

---

## num_train_epochs = 5

| Epoch | Accuracy | F1-score |
|------:|---------:|---------:|
|1|32.73%|27.88%|
|2|59.09%|58.82%|
|3|66.36%|64.43%|
|4|67.27%|66.29%|
|5|67.27%|66.16%|

### Analysis

- Epoch 1 → 3에서 성능이 크게 향상됨.
- Epoch 4에서 Accuracy와 F1-score가 가장 높게 나타남.
- Epoch 5에서는 Accuracy는 동일하지만 F1-score가 소폭 감소하여 성능 향상이 정체되는 모습을 보임.

---

## num_train_epochs = 7

| Epoch | Accuracy | F1-score |
|------:|---------:|---------:|
|1|35.45%|29.79%|
|2|59.09%|58.96%|
|3|66.36%|65.18%|
|4|68.18%|67.67%|
|5|69.09%|68.57%|
|6|69.09%|68.67%|
|7|70.00%|69.80%|

### Analysis

- Epoch가 증가할수록 Accuracy와 F1-score가 꾸준히 향상됨.
- Epoch 7에서 Accuracy와 F1-score 모두 최고 성능을 기록함.

---

# 🏆 Final Result

| Metric | Score |
|---------|------:|
| Accuracy | **70.00%** |
| F1-score | **69.80%** |
| Best Hyperparameter | **num_train_epochs = 7** |

### Conclusion

- 하이퍼파라미터인 **num_train_epochs**를 3, 5, 7로 변경하여 성능을 비교한 결과, **num_train_epochs = 7**에서 가장 높은 Accuracy(70.00%)와 F1-score(69.80%)를 기록하였다.
- 일반적으로 Epoch를 지나치게 증가시키면 과적합이 발생할 수 있지만, 이번 실험에서는 **Epoch 7까지 Accuracy와 F1-score가 꾸준히 향상되어 과적합의 징후는 확인되지 않았다.**

---

# 💻 Demo

```text
상황을 입력하세요.

오늘 너무 덥다.

🍽️ 추천 음식 TOP3

🥇 빙수 (70.12%)
🥈 칼국수 (18.24%)
🥉 삼계탕 (6.31%)
```

---

# 🛠 Tech Stack

| Category | Stack |
|-----------|-------|
| Language | Python |
| Framework | PyTorch |
| NLP | Hugging Face Transformers |
| Model | KLUE-RoBERTa-small |
| Data | Pandas |
| Evaluation | Accuracy, F1-score |
| Version Control | Git / GitHub |

---

# 📚 What I Learned

- Transformer 기반 Text Classification 모델 구현
- KLUE-RoBERTa-small Fine-tuning
- Bidirectional Multi-Head Self-Attention 이해
- 하이퍼파라미터(Epoch) 성능 비교 및 분석
- Accuracy와 F1-score를 활용한 모델 평가
- Hugging Face Trainer 활용
- Git & GitHub를 통한 프로젝트 관리

---

# 👨‍💻 Author

**정희경**

Soonchunhyang University  
Medical IT Engineering
