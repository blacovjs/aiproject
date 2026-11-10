
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# 페이지 설정
st.set_page_config(
    page_title="서울 월별 평균기온 분석 및 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 월별 평균기온 분석 및 예측")
st.write("1907년부터 2025년까지의 기온 데이터를 분석하고 미래 기온을 예측합니다.")

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv(
        "temperature_forecast_data.csv",
        encoding="utf-8-sig"
    )

    columns = ["연도", "월번호", "평균기온(℃)"]

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=columns)

    df["연도"] = df["연도"].astype(int)
    df["월번호"] = df["월번호"].astype(int)

    df = df[
        (df["연도"] >= 1907) &
        (df["연도"] <= 2025)
    ]

    return df


try:
    df = load_data()

    # 월 선택
    month_names = {
        1: "1월", 2: "2월", 3: "3월",
        4: "4월", 5: "5월", 6: "6월",
        7: "7월", 8: "8월", 9: "9월",
        10: "10월", 11: "11월", 12: "12월"
    }

    col1, col2 = st.columns(2)

    with col1:
        selected_month = st.selectbox(
            "분석할 월 선택",
            list(month_names.keys()),
            format_func=lambda x: month_names[x]
        )

    with col2:
        forecast_year = st.number_input(
            "예측할 연도 입력",
            min_value=2026,
            max_value=2100,
            value=2035,
            step=1
        )

    # 선택한 월의 평균기온
    monthly = (
        df[df["월번호"] == selected_month]
        .groupby("연도", as_index=False)["평균기온(℃)"]
        .mean()
        .sort_values("연도")
    )

    if len(monthly) < 10:
        st.warning("분석할 데이터가 충분하지 않습니다.")
        st.stop()

    X = monthly[["연도"]].values
    y = monthly["평균기온(℃)"].values

    # 예측 알고리즘 3개
    models = {
        "선형회귀": LinearRegression(),

        "랜덤포레스트": RandomForestRegressor(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42
        ),

        "그래디언트 부스팅": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            random_state=42
        )
    }

    # 예측할 연도 범위
    first_year = int(monthly["연도"].min())

    all_years = np.arange(
        first_year,
        int(forecast_year) + 1
    )

    X_all = all_years.reshape(-1, 1)

    # 그래프 색상
    colors = {
        "선형회귀": "#E63946",
        "랜덤포레스트": "#2563EB",
        "그래디언트 부스팅": "#16A34A"
    }

    fig = go.Figure()

    # 실제 평균기온
    fig.add_trace(go.Scatter(
        x=monthly["연도"],
        y=monthly["평균기온(℃)"],
        mode="lines+markers",
        name="실제 평균기온",
        line=dict(color="#222222", width=2),
        marker=dict(size=5)
    ))

    results = []

    # 알고리즘별 예측
    for name, model in models.items():

        model.fit(X, y)

        predictions = model.predict(X_all)

        # 과거 학습 구간
        past_mask = all_years <= 2025

        fig.add_trace(go.Scatter(
            x=all_years[past_mask],
            y=predictions[past_mask],
            mode="lines",
            name=name + " 학습 추세",
            line=dict(
                color=colors[name],
                width=2
            )
        ))

        # 미래 예측 구간
        future_mask = all_years >= 2025

        fig.add_trace(go.Scatter(
            x=all_years[future_mask],
            y=predictions[future_mask],
            mode="lines",
            name=name + " 미래 예측",
            line=dict(
                color=colors[name],
                width=3,
                dash="dash"
            )
        ))

        # 입력한 연도의 예측값
        prediction = model.predict(
            np.array([[int(forecast_year)]])
        )[0]

        results.append({
            "알고리즘": name,
            "예측 연도": int(forecast_year),
            "예측 평균기온(℃)": round(
                float(prediction), 2
            )
        })

    # 2025년 기준선
    fig.add_vline(
        x=2025,
        line_dash="dot",
        line_color="gray",
        annotation_text="실제 데이터 마지막 연도"
    )

    # 그래프 설정
    fig.update_layout(
        title="월별 평균기온 분석 및 알고리즘별 예측",
        xaxis_title="연도",
        yaxis_title="평균기온 (℃)",
        hovermode="x unified",
        legend_title="알고리즘",
        template="plotly_white",
        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # 예측 결과
    st.subheader("🔮 알고리즘별 미래 기온 예측")

    result_df = pd.DataFrame(results)

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    # 알고리즘 설명
    st.subheader("🧠 예측 알고리즘 설명")

    with st.expander("1. 선형회귀 (Linear Regression)"):

        st.write("""
        과거 연도와 평균기온 사이에 가장 잘 맞는
        직선을 찾아 미래 기온을 예측합니다.

        - 그래프: 직선 형태
        - 장점: 단순하고 이해하기 쉬움
        - 단점: 복잡한 기온 변화를 표현하기 어려움
        """)

    with st.expander("2. 랜덤포레스트 (Random Forest)"):

        st.write("""
        여러 개의 결정나무를 만들어 각각 예측한 뒤
        그 결과를 평균 내는 방식입니다.

        - 그래프: 계단형 변화가 나타날 수 있음
        - 장점: 비선형적인 패턴 학습 가능
        - 단점: 학습 범위 밖의 미래 추세 예측에 한계가 있음
        """)

    with st.expander("3. 그래디언트 부스팅 (Gradient Boosting)"):

        st.write("""
        이전 모델이 틀린 부분을 다음 모델이 보완하면서
        순서대로 학습하는 방식입니다.

        - 그래프: 구간별 변화 패턴 표현 가능
        - 장점: 복잡한 패턴 학습 가능
        - 단점: 학습 데이터와 설정에 영향을 받음
        """)

    # 주의사항
    st.info("""
    ⚠️ 이 결과는 과거 월별 평균기온과 연도만 이용한
    통계적 예측 실험입니다.

    온실가스, 해양, 대기 순환 등 기후 요인을 반영하지
    않았으므로 공식적인 기후 전망과는 다릅니다.
    """)

    st.caption(
        f"분석 기간: {first_year}~2025년 | "
        f"선택 월: {month_names[selected_month]} | "
        f"관측 연도: {len(monthly)}개"
    )

except FileNotFoundError:
    st.error(
        "temperature_forecast_data.csv 파일이 없습니다. "
        "app.py와 같은 폴더에 업로드하세요."
    )

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")
