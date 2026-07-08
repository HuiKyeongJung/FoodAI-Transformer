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
        "달달하고 시원한 디저트가 필요한 순간, 빙수가 잘 어울려요.",
        "기분 전환이 필요할 때는 차가운 빙수 한 그릇도 좋아요.",
        "무더운 날엔 입안까지 시원해지는 빙수를 추천해요.",
        "오늘처럼 더운 날에는 가볍고 달콤한 빙수가 딱이에요."
    ],
    "칼국수": [
        "춥거나 비 오는 날에는 따뜻한 국물의 칼국수가 잘 어울려요.",
        "든든한 한 끼가 필요할 때는 칼국수를 추천해요.",
        "몸이 으슬으슬한 날엔 따뜻한 칼국수 한 그릇이 좋아요.",
        "구수하고 편안한 국물이 생각난다면 칼국수가 잘 맞아요.",
        "오늘은 속을 따뜻하게 채워줄 칼국수를 추천해요."
    ],
    "삼계탕": [
        "피곤하고 기운이 없을 때는 든든한 삼계탕이 좋아요.",
        "몸보신이 필요한 날에는 따뜻한 삼계탕을 추천해요.",
        "기력이 떨어진 것 같다면 삼계탕으로 든든하게 챙겨보세요.",
        "오늘은 몸을 생각해서 보양식 느낌의 삼계탕이 잘 어울려요.",
        "지친 하루에는 따뜻하고 깊은 맛의 삼계탕을 추천해요."
    ],
    "파전": [
        "비 오는 날에는 바삭한 파전이 딱 잘 어울려요.",
        "흐린 날씨에는 고소한 파전으로 기분을 바꿔보세요.",
        "비 오는 분위기를 즐기고 싶다면 파전을 추천해요.",
        "겉은 바삭하고 속은 촉촉한 파전이 오늘 상황에 잘 맞아요.",
        "날씨가 꿀꿀할 땐 고소한 파전으로 분위기를 살려보세요."
    ],
    "치킨": [
        "시험 끝난 날이나 축하할 일이 있을 땐 치킨이 최고예요.",
        "친구들과 즐겁게 먹기 좋은 음식으로 치킨을 추천해요.",
        "오늘은 바삭한 치킨으로 기분 좋게 마무리해보세요.",
        "특별한 이유가 없어도 치킨은 언제나 좋은 선택이에요.",
        "스트레스 받은 날엔 바삭하고 짭짤한 치킨이 잘 어울려요."
    ],
    "피자": [
        "여럿이 함께 먹기 좋은 메뉴로 피자를 추천해요.",
        "가볍게 분위기 내고 싶을 때 피자가 잘 어울려요.",
        "친구들과 나눠 먹기 좋은 음식으로 피자를 추천해요.",
        "치즈가 당기는 날에는 따뜻한 피자가 딱이에요.",
        "오늘은 부담 없이 즐기기 좋은 피자를 추천해요."
    ],
    "마라탕": [
        "스트레스가 쌓인 날에는 얼얼한 마라탕으로 기분 전환해보세요.",
        "자극적인 음식이 당길 때 마라탕을 추천해요.",
        "매콤하고 얼얼한 맛이 필요한 날엔 마라탕이 잘 어울려요.",
        "오늘은 강한 맛으로 기분을 바꿔줄 마라탕을 추천해요.",
        "입맛이 확 살아나는 메뉴를 원한다면 마라탕이 좋아요."
    ],
    "소고기": [
        "월급날이나 특별한 날에는 소고기로 제대로 즐겨보세요.",
        "기분 좋은 날에는 맛있는 소고기로 스스로를 챙겨보세요.",
        "오늘 하루 고생한 나에게 소고기 한 끼를 선물해보세요.",
        "조금 특별한 식사를 하고 싶다면 소고기를 추천해요.",
        "든든하고 맛있는 한 끼가 필요할 때 소고기가 잘 어울려요."
    ],
    "죽": [
        "속이 불편하거나 입맛이 없을 때는 부드러운 죽이 좋아요.",
        "가볍고 편하게 먹고 싶을 때 죽을 추천해요.",
        "몸 상태가 좋지 않은 날에는 부담 없는 죽이 잘 맞아요.",
        "속을 편하게 달래고 싶다면 따뜻한 죽을 추천해요.",
        "오늘은 자극적이지 않고 부드러운 죽으로 챙겨보세요."
    ],
    "김밥": [
        "간단하게 먹고 싶을 때는 든든한 김밥이 잘 어울려요.",
        "바쁜 날에는 빠르고 간편한 김밥을 추천해요.",
        "가볍지만 든든한 한 끼가 필요하다면 김밥이 좋아요.",
        "이동 중에도 편하게 먹기 좋은 김밥을 추천해요.",
        "간편하면서도 여러 재료를 즐길 수 있는 김밥이 잘 맞아요."
    ],
    "떡볶이": [
        "매콤한 음식이 당길 때는 떡볶이가 딱이에요.",
        "분식이 생각나는 날에는 떡볶이를 추천해요.",
        "달달하고 매콤한 맛이 필요하다면 떡볶이가 잘 어울려요.",
        "기분 전환이 필요한 날에는 매콤한 떡볶이를 추천해요.",
        "간식처럼 먹어도 좋고 한 끼로도 좋은 떡볶이가 잘 맞아요."
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