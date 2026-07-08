import pandas as pd
import torch
import pickle

#정확도 학습 및 평가 과정에서 계산하는 것???
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# 데이터 불러오기
df = pd.read_csv("data.csv")

# label을 숫자로 변환
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["label"])

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# train/test 분리
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 가벼운 한국어 Transformer 모델
model_name = "klue/roberta-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label_id", "labels")
test_dataset = test_dataset.rename_column("label_id", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

num_labels = len(label_encoder.classes_)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

training_args = TrainingArguments(
    output_dir="./food_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=7, #여기가 하이퍼파라미터 (3, 5, 7로 수정하면서 실험하기)
    logging_dir="./logs"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# 최종 평가 결과 출력
eval_result = trainer.evaluate()
print("평가 결과:", eval_result)

model.save_pretrained("./food_model")
tokenizer.save_pretrained("./food_model")

print("FoodAI 학습 완료!")
print("음식 라벨:", list(label_encoder.classes_))