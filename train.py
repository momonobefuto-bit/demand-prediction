import pandas as pd
from xgboost import XGBRegressor

df = pd.read_csv("data/sales.csv")

# 日付を読み込む
df["date"] = pd.to_datetime(df["date"])

# 天気を数値に変換
weather_map = {
    "sunny": 0,
    "cloudy": 1,
    "rainy": 2
}
df["weather"] = df["weather"].map(weather_map)

# 給料日前後3日間なら1、それ以外なら0
def is_payday_period(date):
    day = date.day
    return int(
        12 <= day <= 18
        or 22 <= day <= 28
    )

df["payday_period"] = df["date"].apply(
    is_payday_period
)

# ① 購買利用者数予測AI
X_visitor = df[
    ["weather", "temp", "payday_period"]
]
y_visitor = df["visitors"]

visitor_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

visitor_model.fit(X_visitor, y_visitor)
visitor_model.save_model(
    "models/visitor_model.json"
)
print("利用者数AI：学習完了")

# ② パン別需要予測AI
bread_types = [
    "Aパン", "Bパン", "Cパン", "Dパン"
]

X_bread = df[
    ["weather", "temp", "visitors",
     "payday_period"]
]

for bread in bread_types:
    y_bread = df[bread]

    bread_model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    bread_model.fit(X_bread, y_bread)
    bread_model.save_model(
        f"models/{bread}_model.json"
    )
    print(f"{bread}AI：学習完了")

print("==============================")
print("給料日要素を含む学習が完了しました！")
print("==============================")