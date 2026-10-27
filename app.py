import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from datetime import date


# ==================================
# ページ設定
# ==================================

st.set_page_config(
    page_title="需要予測システム",
    page_icon="🥐"
)


# ==================================
# タイトル
# ==================================

st.title("需要予測システム")

st.write(
    "天気と気温から購買の利用者数とパンの需要を予測します。"
)


# ==================================
# 入力
# ==================================

st.header("予測条件")


weather_name = st.selectbox(
    "天気",
    ["晴れ", "曇り", "雨"]
)


temp = st.number_input(
    "気温（℃）",
    min_value=-10.0,
    max_value=40.0,
    value=25.0,
    step=0.5
)
prediction_date = st.date_input(
    "予測日",
    value=date.today()
)

payday_period = int(
    12 <= prediction_date.day <= 18
    or 22 <= prediction_date.day <= 28
)

# 天気を数字に変換
weather_map = {
    "晴れ": 0,
    "曇り": 1,
    "雨": 2
}

weather = weather_map[weather_name]


# ==================================
# 予測ボタン
# ==================================

if st.button("需要を予測する"):

    # --------------------------------
    # 利用者数を予測
    # --------------------------------

    visitor_data = pd.DataFrame([{
    "weather": weather,
    "temp": temp,
    "payday_period": payday_period
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


    # --------------------------------
    # 利用者数を表示
    # --------------------------------

    st.subheader("👥 予測利用者数")

    st.metric(
        "予測利用者数",
        f"{predicted_visitors}人"
    )


    # --------------------------------
    # パン需要予測
    # --------------------------------

    bread_data = pd.DataFrame([{
    "weather": weather,
    "temp": temp,
    "visitors": predicted_visitors,
    "payday_period": payday_period
}])


    prices = {
        "Aパン": 130,
        "Bパン": 150,
        "Cパン": 180,
        "Dパン": 230
    }


    results = []

    total_sales = 0


    st.subheader("パン別需要予測")


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


        results.append({
            "商品": bread,
            "予測販売数": quantity,
            "予測売上": sales
        })


    # --------------------------------
    # 結果表示
    # --------------------------------

    result_df = pd.DataFrame(results)


    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------
    # 合計売上
    # --------------------------------

    st.subheader("予測売上")


    st.metric(
        "予測売上合計",
        f"{total_sales:,}円"
    )


    # --------------------------------
    # グラフ
    # --------------------------------

    st.subheader("📊 パン別需要")


    chart_df = result_df.set_index("商品")[
        ["予測販売数"]
    ]


    st.bar_chart(chart_df)