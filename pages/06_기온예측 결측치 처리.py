import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(
    page_title="월별 기온 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 기온 예측 및 결측치 처리")
st.caption("과거 기온 데이터를 분석하고 3가지 머신러닝 알고리즘으로 미래 기온을 비교합니다.")

# pages 폴더의 상위 폴더에 있는 temperature.csv 파일
DATA_FILE = Path(__file__).resolve().parent.parent / "temperature.csv"

month_names = {
    1: "1월", 2: "2월", 3: "3월", 4: "4월",
    5: "5월", 6: "6월", 7: "7월", 8: "8월",
    9: "9월", 10: "10월", 11: "11월", 12: "12월"
}

if not DATA_FILE.exists():
    st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_FILE}")
    st.stop()

try:
    # CSV 읽기
    try:
        df = pd.read_csv(DATA_FILE, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    # 컬럼명 공백 제거
    df.columns = df.columns.astype(str).str.strip()

    # 날짜 컬럼 확인
    if "일시" not in df.columns:
        st.error("CSV 파일에 '일시' 컬럼이 없습니다.")
        st.stop()

    # 연도와 월번호가 없는 경우 일시에서 추출
    if "연도" not in df.columns or "월번호" not in df.columns:
        date_parts = df["일시"].astype(str).str.extract(
            r"([A-Za-z]{3})-(\d{2,4})"
        )

        month_map = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
            "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
            "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
        }

        df["월번호"] = date_parts[0].map(month_map)

        years = pd.to_numeric(date_parts[1], errors="coerce")

        # 2자리 연도 변환
        if years.notna().any() and years.max() <= 99:
            years = years.apply(
                lambda y: 1900 + y if pd.notna(y) and y >= 30
                else 2000 + y if pd.notna(y) else np.nan
            )

        df["연도"] = years

    # 숫자형 변환
    numeric_columns = [
        "연도", "월번호", "평균기온(℃)",
        "평균최고기온(℃)", "최고기온(℃)",
        "평균최저기온(℃)", "최저기온(℃)"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "평균기온(℃)" not in df.columns:
        st.error("CSV 파일에 '평균기온(℃)' 컬럼이 없습니다.")
        st.stop()

    # 결측치 처리 전 데이터 확인
    st.subheader("📋 데이터 결측치 확인")

    missing_count = df.isnull().sum()

    missing_df = pd.DataFrame({
        "컬럼명": missing_count.index,
        "결측치 개수": missing_count.values
    })

    st.dataframe(missing_df, use_container_width=True)

    # 기온 결측치 처리
    temperature_columns = [
        col for col in [
            "평균기온(℃)", "평균최고기온(℃)",
            "최고기온(℃)", "평균최저기온(℃)",
            "최저기온(℃)"
        ] if col in df.columns
    ]

    df = df.sort_values(["연도", "월번호"])

    for col in temperature_columns:
        df[col] = df[col].interpolate(
            method="linear",
            limit_direction="both"
        )

    st.success("기온 결측치 처리가 완료되었습니다.")

    st.divider()

    # 월 선택
    st.subheader("📅 분석할 월 선택")

    selected_month = st.selectbox(
        "월을 선택하세요.",
        options=list(month_names.keys()),
        format_func=lambda x: month_names[x]
    )

    # 선택 월 데이터
    monthly = df[
        (df["월번호"] == selected_month) &
        (df["연도"].between(1907, 2025))
    ][["연도", "평균기온(℃)"]].dropna()

    monthly = monthly.groupby("연도", as_index=False)["평균기온(℃)"].mean()
    monthly = monthly.sort_values("연도")

    if len(monthly) < 5:
        st.warning("선택한 월의 데이터가 부족합니다.")
        st.stop()

    # 예측 연도
    st.subheader("🔮 미래 기온 예측")

    future_year = st.number_input(
        "예측할 연도",
        min_value=2026,
        max_value=2100,
        value=2030,
        step=1
    )

    X = monthly[["연도"]].values
    y = monthly["평균기온(℃)"].values

    future_years = np.arange(
        int(monthly["연도"].min()),
        int(future_year) + 1
    ).reshape(-1, 1)

    # 알고리즘 3개
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

    colors = {
        "선형회귀": "#3498db",
        "랜덤포레스트": "#e67e22",
        "그래디언트 부스팅": "#2ecc71"
    }

    fig = go.Figure()

    # 실제 관측 데이터
    fig.add_trace(go.Scatter(
        x=monthly["연도"],
        y=monthly["평균기온(℃)"],
        mode="lines+markers",
        name="실제 평균기온",
        line=dict(color="black", width=2),
        marker=dict(size=5)
    ))

    prediction_results = []

    for name, model in models.items():
        model.fit(X, y)

        predictions = model.predict(future_years)

        # 학습 구간 그래프
        historical_predictions = model.predict(X)

        fig.add_trace(go.Scatter(
            x=monthly["연도"],
            y=historical_predictions,
            mode="lines",
            name=f"{name} 학습 결과",
            line=dict(color=colors[name], width=2)
        ))

        # 미래 예측 그래프
        fig.add_trace(go.Scatter(
            x=future_years.flatten(),
            y=predictions,
            mode="lines",
            name=f"{name} 미래 예측",
            line=dict(color=colors[name], width=3, dash="dash")
        ))

        prediction_value = float(
            model.predict(np.array([[future_year]]))[0]
        )

        prediction_results.append({
            "알고리즘": name,
            "예측 연도": int(future_year),
            "예측 평균기온(℃)": round(prediction_value, 2)
        })

    fig.add_vline(
        x=2025,
        line_dash="dot",
        line_color="gray",
        annotation_text="예측 시작 구간"
    )

    fig.update_layout(
        title=f"{month_names[selected_month]} 연도별 평균기온 및 미래 예측",
        xaxis_title="연도",
        yaxis_title="평균기온(℃)",
        hovermode="x unified",
        template="plotly_white",
        legend_title="데이터 및 알고리즘",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # 예측 결과
    st.subheader(f"📊 {future_year}년 예측 결과")

    result_df = pd.DataFrame(prediction_results)
    st.dataframe(result_df, use_container_width=True)

    # 알고리즘 설명
    st.subheader("🧠 예측 알고리즘 설명")

    with st.expander("1. 선형회귀 (Linear Regression)"):
        st.write(
            "과거 연도와 평균기온 사이의 직선 관계를 학습합니다. "
            "전체적인 상승 또는 하락 추세를 직선으로 표현합니다."
        )

    with st.expander("2. 랜덤포레스트 (Random Forest)"):
        st.write(
            "여러 결정나무가 각각 예측한 값을 평균 내는 방식입니다. "
            "과거 데이터의 구간별 패턴을 학습하므로 계단형 변화가 나타날 수 있습니다. "
            "학습 범위를 벗어난 미래에서는 추세 변화를 잘 예측하지 못할 수 있습니다."
        )

    with st.expander("3. 그래디언트 부스팅 (Gradient Boosting)"):
        st.write(
            "앞선 나무 모델의 오차를 다음 나무가 보완하도록 순서대로 학습합니다. "
            "작은 변화와 비선형 패턴을 단계적으로 학습합니다. "
            "결과는 학습 데이터와 모델 설정에 영향을 받습니다."
        )

    st.info(
        "주의: 이 앱은 과거 월별 평균기온과 연도만 사용하는 비교용 예측입니다. "
        "온실가스, 해양, 대기 순환 등 기후 요인을 반영한 공식 기후 전망이 아닙니다."
    )

    st.divider()

    # 결측치 처리 결과 다운로드
    st.subheader("💾 결측치 처리 데이터 다운로드")

    csv_data = df.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        label="처리된 CSV 다운로드",
        data=csv_data,
        file_name="temperature_cleaned.csv",
        mime="text/csv"
    )

except Exception as error:
    st.error(f"데이터 처리 중 오류가 발생했습니다: {error}")
