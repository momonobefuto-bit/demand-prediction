import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from matplotlib import font_manager

# ==================================
# 日本語フォントを設定
# ==================================

font_path = r"C:\Windows\Fonts\meiryo.ttc"

font_prop = font_manager.FontProperties(
    fname=font_path
)

plt.rcParams["font.family"] = font_prop.get_name()


# ==================================
# 需要予測システム
# ==================================

print("================================")
print("      需要予測システム")
print("================================")


# ==================================
# ① 天気を入力
# ==================================

while True:

    print("\n天気を選んでください")
    print("1：晴れ")
    print("2：曇り")
    print("3：雨")

    try:
        weather_input = int(input(">> "))

        if weather_input in [1, 2, 3]:
            break

        print("1〜3の数字を入力してください！")

    except ValueError:
        print("数字を入力してください！")


weather_map = {
    1: 0,
    2: 1,
    3: 2
}

weather = weather_map[weather_input]


# ==================================
# ② 気温を入力
# ==================================

while True:

    try:
        temp = float(
            input("\n気温を入力してください（℃）\n>> ")
        )

        if -10 <= temp <= 40:
            break

        print("-10〜40℃の範囲で入力してください！")

    except ValueError:
        print("数字を入力してください！")


# ==================================
# ③ 利用者数をAIで予測
# ==================================

visitor_data = pd.DataFrame([{
    "weather": weather,
    "temp": temp
}])


visitor_model = XGBRegressor()

visitor_model.load_model(
    "models/visitor_model.json"
)
visitor_prediction = visitor_model.predict(
    visitor_data
)


predicted_visitors = round(
    visitor_prediction[0]
)

# 今回のデータ範囲に合わせる
predicted_visitors = max(
    50,
    min(100, predicted_visitors)
)


print("\n================================")
print("        AI予測結果")
print("================================")

print(f"予測利用者数：{predicted_visitors}人")


# ==================================
# ④ パンの需要をAIで予測
# ==================================

bread_data = pd.DataFrame([{
    "weather": weather,
    "temp": temp,
    "visitors": predicted_visitors
}])


prices = {
    "Aパン": 130,
    "Bパン": 150,
    "Cパン": 180,
    "Dパン": 230
}


bread_names = []
quantities = []

total_sales = 0


print("\n--- パン別需要予測 ---")


for bread, price in prices.items():

    model = XGBRegressor()

    model.load_model(
        f"models/{bread}_model.json"
    )

    prediction = model.predict(
        bread_data
    )

    quantity = max(
        0,
        round(prediction[0])
    )

    sales = quantity * price

    total_sales += sales

    bread_names.append(bread)
    quantities.append(quantity)

    print(f"{bread}：{quantity}個")
    print(f"　　 予測売上：{sales}円")


# ==================================
# ⑤ 合計売上
# ==================================

print("\n================================")
print(f"予測売上合計：{total_sales}円")
print("================================")


# ==================================
# ⑥ パン別グラフ
# ==================================

plt.figure(figsize=(8, 5))

plt.bar(
    bread_names,
    quantities
)

plt.title(
    "パン別 需要予測",
    fontproperties=font_prop
)

plt.xlabel(
    "商品",
    fontproperties=font_prop
)

plt.ylabel(
    "予測販売数（個）",
    fontproperties=font_prop
)

plt.xticks(
    fontproperties=font_prop
)

plt.tight_layout()

plt.show()