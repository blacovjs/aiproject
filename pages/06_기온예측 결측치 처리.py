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
st.caption("결측치 처리 방법과 머신러닝 알고리즘에 따라 달라지는 기온 예측 결과를 비교합니다.")

# CSV 파일 경로
DATA_FILE = Path(__file__).resolve().parent.parent / "temperature.csv"

month_names = {
    1: "1월", 2: "2월", 3: "3월", 4: "4월",
    5: "5월", 6: "6월", 7: "7월", 8: "8월",
    9: "9월", 10: "10월", 11: "11월", 12: "12월"
}

if not DATA_FILE.exists():
    st.error("temperature.csv 파일을 찾을 수 없습니다. 프로젝트 최상위 폴더에 파일을 넣어주세요.")
    st.stop()

try:
    # CSV 불러오기
    try:
        df = pd.read_csv(DATA_FILE, encoding="cp949")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    df.columns = df.columns.astype(str).str.strip()

    if "일시" not in df.columns:
        st.error("CSV 파일에 '일시' 컬럼이 없습니다.")
        st.stop()

    # 연도와 월 정보 확인 및 추출
    if "연도" not in df.columns:
        df["연도"] = np.nan

    if "월번호" not in df.columns:
        df["월번호"] = np.nan

    # 일시에서 연도와 월 추출
    date_text = df["일시"].astype(str).str.strip()

    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    extracted = date_text.str.extract(
        r"(?i)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[- /](\d{2,4})"
    )

    extracted_month = extracted[0].str.title().map(month_map)
    extracted_year = pd.to_numeric(extracted[1], errors="coerce")

    # 기존 연도와 월번호가 없는 경우에만 일시에서 추출
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce")
    df["월번호"] = pd.to_numeric(df["월번호"], errors="coerce")

    df["연도"] = df["연도"].fillna(extracted_year)
    df["월번호"] = df["월번호"].fillna(extracted_month)

    # 연도와 월번호 숫자형 변환
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce")
    df["월번호"] = pd.to_numeric(df["월번호"], errors="coerce")

    # 날짜 정보가 없는 행 제거
    df = df.dropna(subset=["연도", "월번호"])
    df["연도"] = df["연도"].astype(int)
    df["월번호"] = df["월번호"].astype(int)

    # 기온 컬럼 숫자형 변환
    temperature_columns = [
        col for col in [
            "평균기온(℃)",
            "평균최고기온(℃)",
            "최고기온(℃)",
            "평균최저기온(℃)",
            "최저기온(℃)"
        ] if col in df.columns
    ]

    if "평균기온(℃)" not in df.columns:
        st.error("CSV 파일에 '평균기온(℃)' 컬럼이 없습니다.")
        st.stop()

    for col in temperature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["연도", "월번호"]).reset_index(drop=True)

    # 원본 데이터 보관
    original_df = df.copy()

    st.subheader("📋 원본 데이터 결측치 확인")

    missing_count = original_df.isnull().sum()

    missing_df = pd.DataFrame({
        "컬럼명": missing_count.index,
        "결측치 개수": missing_count.values
    })

    st.dataframe(missing_df, use_container_width=True)

    st.divider()

    # 결측치 처리 방법 선택
    st.subheader("1️⃣ 결측치 처리 방법 선택")

    missing_method = st.selectbox(
        "결측치를 어떤 방법으로 처리할까요?",
        [
            "결측치 제거",
            "중앙값으로 대체",
            "선형보간법"
        ]
    )

    method_description = {
        "결측치 제거": "평균기온이 비어 있는 행을 제거하고 남은 데이터로 예측합니다.",
        "중앙값으로 대체": "같은 월에 해당하는 기온 데이터의 중앙값으로 결측치를 채웁니다.",
        "선형보간법": "같은 월의 이전·이후 연도 데이터를 이용해 결측치를 직선 형태로 추정합니다."
    }

    st.info(method_description[missing_method])

    # 결측치 처리 실행
    processed_df = original_df.copy()

    if missing_method == "결측치 제거":
        processed_df = processed_df.dropna(
            subset=["평균기온(℃)"]
        )

    elif missing_method == "중앙값으로 대체":
        for col in temperature_columns:
            processed_df[col] = processed_df.groupby("월번호")[col].transform(
                lambda s: s.fillna(s.median())
            )

    elif missing_method == "선형보간법":
        for col in temperature_columns:
            processed_df[col] = processed_df.groupby("월번호")[col].transform(
                lambda s: s.interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )

    processed_df = processed_df.sort_values(
        ["연도", "월번호"]
    ).reset_index(drop=True)

    # 처리 결과 확인
    remaining_missing = processed_df["평균기온(℃)"].isnull().sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("원본 데이터 행 수", f"{len(original_df):,}개")

    with col2:
        st.metric("처리 후 데이터 행 수", f"{len(processed_df):,}개")

    with col3:
        st.metric("남은 평균기온 결측치", f"{remaining_missing:,}개")

    st.success(f"{missing_method} 처리가 완료되었습니다.")

    st.divider()

    # 분석 조건 선택
    st.subheader("2️⃣ 예측 조건 선택")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_month = st.selectbox(
            "분석할 월",
            options=list(month_names.keys()),
            format_func=lambda x: month_names[x]
        )

    with col2:
        selected_algorithm = st.selectbox(
            "예측 알고리즘",
            [
                "선형회귀",
                "랜덤포레스트",
                "그래디언트 부스팅"
            ]
        )

    with col3:
        future_year = st.number_input(
            "예측할 연도",
            min_value=2026,
            max_value=2100,
            value=2030,
            step=1
        )

    # 선택한 월 데이터 추출
    monthly = processed_df[
        (processed_df["월번호"] == selected_month) &
        (processed_df["연도"].between(1907, 2025))
    ][["연도", "평균기온(℃)"]].dropna()

    monthly = monthly.groupby(
        "연도", as_index=False
    )["평균기온(℃)"].mean()

    monthly = monthly.sort_values("연도")

    if len(monthly) < 5:
        st.warning("선택한 월의 데이터가 부족하여 예측할 수 없습니다.")
        st.stop()

    X = monthly[["연도"]].values
    y = monthly["평균기온(℃)"].values

    # 알고리즘 선택
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

    # 선택 알고리즘 학습
    model = models[selected_algorithm]
    model.fit(X, y)

    future_years = np.arange(
        int(monthly["연도"].min()),
        int(future_year) + 1
    ).reshape(-1, 1)

    predictions = model.predict(future_years)

    future_prediction = float(
        model.predict(np.array([[int(future_year)]]))[0]
    )

    st.divider()

    # 예측 결과
    st.subheader("3️⃣ 선택한 조건의 기온 예측 결과")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("결측치 처리 방법", missing_method)

    with col2:
        st.metric("선택 알고리즘", selected_algorithm)

    with col3:
        st.metric(
            f"{future_year}년 예측 평균기온",
            f"{future_prediction:.2f}℃"
        )

    # 그래프
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly["연도"],
        y=monthly["평균기온(℃)"],
        mode="lines+markers",
        name="실제 평균기온",
        line=dict(color="black", width=2),
        marker=dict(size=5)
    ))

    fig.add_trace(go.Scatter(
        x=future_years.flatten(),
        y=predictions,
        mode="lines",
        name=f"{selected_algorithm} 예측",
        line=dict(
            color=colors[selected_algorithm],
            width=3,
            dash="dash"
        )
    ))

    fig.add_vline(
        x=2025,
        line_dash="dot",
        line_color="gray",
        annotation_text="예측 구간"
    )

    fig.update_layout(
        title=f"{month_names[selected_month]} 기온 예측",
        xaxis_title="연도",
        yaxis_title="평균기온(℃)",
        hovermode="x unified",
        template="plotly_white",
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 결측치 처리 방법별 예측 비교
    st.subheader("4️⃣ 결측치 처리 방법에 따른 예측값 비교")

    comparison_results = []

    methods = [
        "결측치 제거",
        "중앙값으로 대체",
        "선형보간법"
    ]

    for method in methods:
        temp_df = original_df.copy()

        if method == "결측치 제거":
            temp_df = temp_df.dropna(
                subset=["평균기온(℃)"]
            )

        elif method == "중앙값으로 대체":
            for col in temperature_columns:
                temp_df[col] = temp_df.groupby("월번호")[col].transform(
                    lambda s: s.fillna(s.median())
                )

        elif method == "선형보간법":
            for col in temperature_columns:
                temp_df[col] = temp_df.groupby("월번호")[col].transform(
                    lambda s: s.interpolate(
                        method="linear",
                        limit_direction="both"
                    )
                )

        temp_monthly = temp_df[
            (temp_df["월번호"] == selected_month) &
            (temp_df["연도"].between(1907, 2025))
        ][["연도", "평균기온(℃)"]].dropna()

        temp_monthly = temp_monthly.groupby(
            "연도", as_index=False
        )["평균기온(℃)"].mean()

        temp_monthly = temp_monthly.sort_values("연도")

        if len(temp_monthly) < 5:
            continue

        temp_X = temp_monthly[["연도"]].values
        temp_y = temp_monthly["평균기온(℃)"].values

        # 선택한 알고리즘으로 각 결측치 처리 방법의 예측
        comparison_model = models[selected_algorithm]
        comparison_model.fit(temp_X, temp_y)

        result = float(
            comparison_model.predict(
                np.array([[int(future_year)]])
            )[0]
        )

        comparison_results.append({
            "결측치 처리 방법": method,
            "예측 평균기온(℃)": round(result, 2)
        })

    comparison_df = pd.DataFrame(comparison_results)

    if not comparison_df.empty:
        comparison_fig = go.Figure()

        comparison_fig.add_trace(go.Bar(
            x=comparison_df["결측치 처리 방법"],
            y=comparison_df["예측 평균기온(℃)"],
            text=comparison_df["예측 평균기온(℃)"].apply(
                lambda x: f"{x:.2f}℃"
            ),
            textposition="outside",
            marker_color=["#3498db", "#e67e22", "#2ecc71"]
        ))

        comparison_fig.update_layout(
            title=f"{future_year}년 결측치 처리 방법별 예측 결과",
            xaxis_title="결측치 처리 방법",
            yaxis_title="예측 평균기온(℃)",
            template="plotly_white",
            height=450,
            showlegend=False
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # 알고리즘별 예측 결과 비교
    st.subheader("5️⃣ 알고리즘별 예측값 비교")

    algorithm_results = []

    for algorithm_name, algorithm_model in models.items():
        algorithm_model.fit(X, y)

        result = float(
            algorithm_model.predict(
                np.array([[int(future_year)]])
            )[0]
        )

        algorithm_results.append({
            "알고리즘": algorithm_name,
            "예측 평균기온(℃)": round(result, 2)
        })

    algorithm_df = pd.DataFrame(algorithm_results)

    algorithm_fig = go.Figure()

    algorithm_fig.add_trace(go.Bar(
        x=algorithm_df["알고리즘"],
        y=algorithm_df["예측 평균기온(℃)"],
        text=algorithm_df["예측 평균기온(℃)"].apply(
            lambda x: f"{x:.2f}℃"
        ),
        textposition="outside",
        marker_color=[
            colors[name] for name in algorithm_df["알고리즘"]
        ]
    ))

    algorithm_fig.update_layout(
        title=f"{future_year}년 알고리즘별 기온 예측 비교",
        xaxis_title="예측 알고리즘",
        yaxis_title="예측 평균기온(℃)",
        template="plotly_white",
        height=450,
        showlegend=False
    )

    st.plotly_chart(
        algorithm_fig,
        use_container_width=True
    )

    st.dataframe(
        algorithm_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # 알고리즘 설명
    st.subheader("🧠 예측 알고리즘 설명")

    with st.expander("1. 선형회귀"):
        st.write(
            "과거 연도와 평균기온 사이의 관계를 직선으로 학습합니다. "
            "전체적인 기온 상승 또는 하락 추세를 예측하는 데 사용됩니다."
        )

    with st.expander("2. 랜덤포레스트"):
        st.write(
            "여러 결정나무의 예측 결과를 평균 내는 알고리즘입니다. "
            "데이터의 구간별 패턴을 학습하지만, 학습 범위를 벗어난 미래 예측에서는 "
            "마지막 구간의 값이 유지되는 경향이 있습니다."
        )

    with st.expander("3. 그래디언트 부스팅"):
        st.write(
            "여러 결정나무가 앞선 모델의 오차를 보완하도록 순서대로 학습합니다. "
            "비선형적인 패턴을 학습할 수 있지만, 미래 외삽에는 한계가 있습니다."
        )

    st.info(
        "참고: 결측치 처리 방법과 알고리즘에 따라 예측 결과가 달라질 수 있습니다. "
        "랜덤포레스트와 그래디언트 부스팅은 과거 데이터의 패턴을 학습하는 모델이므로 "
        "미래의 장기적인 기온 상승 추세를 그대로 반영하지 못할 수 있습니다. "
        "이 결과는 공식적인 기후 전망이 아닌 비교·학습용 예측입니다."
    )

    st.divider()

    # 처리된 데이터 다운로드
    st.subheader("💾 결측치 처리 데이터 다운로드")

    download_df = processed_df.copy()

    csv_data = download_df.to_csv(
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
