# import torch
# import pickle
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# # 모델 불러오기
# model_path = "./food_model"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)

# # 라벨 불러오기
# with open("label_encoder.pkl", "rb") as f:
#     label_encoder = pickle.load(f)

# model.eval()

# while True:

#     text = input("\n상황을 입력하세요 (종료:q) : (ex. 더운 날씨에 뭐 먹지?) ")

#     if text.lower() == "q":
#         break

#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=True,
#         max_length=64
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1)[0]

#     top3 = torch.topk(probs, k=3)

#    emoji_map = {
#     "빙수": "🍧",
#     "칼국수": "🍜",
#     "삼계탕": "🥘",
#     "파전": "🥞",
#     "치킨": "🍗",
#     "피자": "🍕",
#     "마라탕": "🌶️",
#     "소고기": "🥩",
#     "죽": "🥣",
#     "김밥": "🍙",
#     "떡볶이": "🔥"
# }

# print("\n🍽️ FoodAI 추천 결과")
# print("━━━━━━━━━━━━━━━━━━━━")

# for rank, (idx, score) in enumerate(zip(top3.indices, top3.values), start=1):

#     food = label_encoder.inverse_transform([idx.item()])[0]
#     emoji = emoji_map.get(food, "🍽️")

#     medal = ["🥇", "🥈", "🥉"][rank - 1]

#     print(f"{medal} {emoji} {food:<6} {score.item()*100:.2f}%")

# print("━━━━━━━━━━━━━━━━━━━━")

import torch
import pickle
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 모델 불러오기
model_path = "./food_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 라벨 불러오기
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model.eval()

emoji_map = {
    "빙수": "🍧",
    "칼국수": "🍜",
    "삼계탕": "🥘",
    "파전": "🥞",
    "치킨": "🍗",
    "피자": "🍕",
    "마라탕": "🌶️",
    "소고기": "🥩",
    "죽": "🥣",
    "김밥": "🍙",
    "떡볶이": "🔥"
}

reason_map = {
    "빙수": [
        "더운 날에는 시원한 빙수로 더위를 날려보세요!",
        "무더위에는 시원하고 달달한 빙수가 잘 어울려요."
    ],
    "칼국수": [
        "비 오는 날에는 따뜻한 국물의 칼국수가 잘 어울려요.",
        "이열치열! 한바탕 땀을 흘리면 몸이 가벼워질지도 몰라요!"
    ],
    "삼계탕": [
        "피곤하고 기운이 없을 때는 든든한 삼계탕이 좋아요.",
        "몸보신이 필요한 날에는 따뜻한 삼계탕을 추천해요."
    ],
    "파전": [
        "비 오는 날에는 바삭한 파전이 딱 잘 어울려요.",
        "흐린 날씨에는 고소한 파전으로 기분을 바꿔보세요."
    ],
    "치킨": [
        "시험 끝난 날이나 축하할 일이 있을 땐 치킨이 최고예요.",
        "친구들과 즐겁게 먹기 좋은 음식으로 치킨을 추천해요."
    ],
    "피자": [
        "여럿이 함께 먹기 좋은 메뉴로 피자를 추천해요.",
        "가볍게 분위기 내고 싶을 때 피자가 잘 어울려요."
    ],
    "마라탕": [
        "스트레스가 쌓인 날에는 얼얼한 마라탕으로 기분 전환해보세요.",
        "자극적인 음식이 당길 때 마라탕을 추천해요."
    ],
    "소고기": [
        "월급날이나 특별한 날에는 소고기로 제대로 즐겨보세요.",
        "기분 좋은 날에는 맛있는 소고기로 스스로를 챙겨보세요."
    ],
    "죽": [
        "속이 불편하거나 입맛이 없을 때는 부드러운 죽이 좋아요.",
        "가볍고 편하게 먹고 싶을 때 죽을 추천해요."
    ],
    "김밥": [
        "간단하게 먹고 싶을 때는 든든한 김밥이 잘 어울려요.",
        "바쁜 날에는 빠르고 간편한 김밥을 추천해요."
    ],
    "떡볶이": [
        "매콤한 음식이 당길 때는 떡볶이가 딱이에요.",
        "분식이 생각나는 날에는 떡볶이를 추천해요."
    ]
}

while True:
    text = input("\n상황을 입력하세요 (종료:q) : (ex. 더운 날씨에 뭐먹지?) ")

    if text.lower() == "q":
        print("FoodAI를 종료합니다.")
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    top3 = torch.topk(probs, k=3)

    print("\n🍽️ FoodAI 추천 결과")
    print("━━━━━━━━━━━━━━━━━━━━")

    top_food = None

    for rank, (idx, score) in enumerate(zip(top3.indices, top3.values), start=1):
        food = label_encoder.inverse_transform([idx.item()])[0]
        emoji = emoji_map.get(food, "🍽️")
        medal = ["🥇", "🥈", "🥉"][rank - 1]

        if rank == 1:
            top_food = food

        print(f"{medal} {emoji} {food:<6} {score.item()*100:.2f}%")

    print("\n💡 추천 문구")
    if top_food in reason_map:
        print(random.choice(reason_map[top_food]))
    else:
        print("오늘 상황에 잘 어울리는 음식을 추천해봤어요!")

    print("━━━━━━━━━━━━━━━━━━━━")