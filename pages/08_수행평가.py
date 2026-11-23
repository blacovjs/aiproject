
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="Penguin AI Lab",
    page_icon="🐧",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
h1 {
    color: #17365d;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# =====================================
# 2. 데이터 불러오기
# =====================================

@st.cache_data
def load_data(uploaded_file=None):

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("penguins.csv")

    required = [
        "species",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    missing_columns = [
        col for col in required if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {missing_columns}"
        )

    data = data[required].copy()

    for col in required[1:]:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data = data.dropna(subset=["species"])

    data["species"] = data["species"].astype(str)

    return data


# =====================================
# 3. 제목
# =====================================

st.title("🐧 Penguin AI Lab")
st.subheader("인공지능 펭귄 종류 판별 프로그램")

st.markdown("""
펭귄의 신체 정보를 분석하여 종류를 예측하는 인공지능 프로그램입니다.

KNN, 의사결정나무, 랜덤 포레스트
3가지 머신러닝 알고리즘을 직접 비교해 보세요!
""")

st.divider()

# =====================================
# 4. 사이드바
# =====================================

st.sidebar.title("⚙️ AI 설정")

uploaded_file = st.sidebar.file_uploader(
    "펭귄 CSV 파일 업로드 (선택)",
    type=["csv"]
)

test_size = st.sidebar.slider(
    "테스트 데이터 비율",
    0.1, 0.4, 0.2, 0.05
)

random_state = st.sidebar.slider(
    "데이터 분할 난수 설정",
    1, 100, 42
)

# =====================================
# 5. 데이터 로딩
# =====================================

try:
    df = load_data(uploaded_file)

except FileNotFoundError:
    st.error(
        "penguins.csv 파일을 찾을 수 없습니다. "
        "GitHub 저장소에 app.py와 같은 위치로 업로드해 주세요."
    )
    st.stop()

except Exception as e:
    st.error(f"데이터 오류: {e}")
    st.stop()

features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]

X = df[features]
y = df["species"]

if y.nunique() < 2:
    st.error("분류하려면 최소 2종류 이상의 펭귄 데이터가 필요합니다.")
    st.stop()

# =====================================
# 6. 데이터 분할
# =====================================

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
except ValueError as e:
    st.error(f"데이터 분할 오류: {e}")
    st.stop()

# =====================================
# 7. 인공지능 알고리즘
# =====================================

models = {

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=5,
            random_state=random_state
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=random_state,
            class_weight="balanced"
        ))
    ])
}

# =====================================
# 8. 모델 학습
# =====================================

results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    trained_models[name] = model

    pred = model.predict(X_test)

    results.append({
        "알고리즘": name,
        "정확도": accuracy_score(y_test, pred),
        "정밀도": precision_score(
            y_test, pred, average="macro", zero_division=0
        ),
        "재현율": recall_score(
            y_test, pred, average="macro", zero_division=0
        ),
        "F1 점수": f1_score(
            y_test, pred, average="macro", zero_division=0
        )
    })

result_df = pd.DataFrame(results)

# =====================================
# 9. 상단 데이터 요약
# =====================================

st.header("📊 데이터 현황")

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 펭귄 데이터", f"{len(df)}마리")
c2.metric("학습 데이터", f"{len(X_train)}마리")
c3.metric("테스트 데이터", f"{len(X_test)}마리")
c4.metric("펭귄 종류", f"{y.nunique()}종")

st.divider()

# =====================================
# 10. 탭
# =====================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🐧 펭귄 예측",
    "🏆 알고리즘 비교",
    "📈 데이터 시각화",
    "🔎 데이터 분석"
])

# =====================================
# TAB 1. 펭귄 예측
# =====================================

with tab1:

    st.header("🐧 새로운 펭귄 종류 예측")

    st.write(
        "펭귄의 신체 정보를 입력하면 "
        "AI가 가장 유사한 종류를 예측합니다."
    )

    col1, col2 = st.columns(2)

    defaults = {
        "bill_length_mm": 43.0,
        "bill_depth_mm": 17.0,
        "flipper_length_mm": 200.0,
        "body_mass_g": 4000.0
    }

    with col1:

        bill_length = st.number_input(
            "부리 길이 (mm)",
            min_value=20.0,
            max_value=70.0,
            value=defaults["bill_length_mm"],
            step=0.1
        )

        bill_depth = st.number_input(
            "부리 깊이 (mm)",
            min_value=10.0,
            max_value=30.0,
            value=defaults["bill_depth_mm"],
            step=0.1
        )

    with col2:

        flipper_length = st.number_input(
            "날개 길이 (mm)",
            min_value=150.0,
            max_value=250.0,
            value=defaults["flipper_length_mm"],
            step=0.1
        )

        body_mass = st.number_input(
            "몸무게 (g)",
            min_value=2000.0,
            max_value=7000.0,
            value=defaults["body_mass_g"],
            step=50.0
        )

    selected_model = st.selectbox(
        "AI 알고리즘 선택",
        list(trained_models.keys())
    )

    if st.button(
        "🔍 펭귄 종류 예측하기",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{
            "bill_length_mm": bill_length,
            "bill_depth_mm": bill_depth,
            "flipper_length_mm": flipper_length,
            "body_mass_g": body_mass
        }])

        model = trained_models[selected_model]

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        classes = model.named_steps["model"].classes_

        st.success(
            f"🎉 예측된 펭귄 종류: {prediction}"
        )

        prob_df = pd.DataFrame({
            "펭귄 종류": classes,
            "예측 확률": probabilities
        })

        fig = px.bar(
            prob_df,
            x="펭귄 종류",
            y="예측 확률",
            color="펭귄 종류",
            text=prob_df["예측 확률"].apply(
                lambda x: f"{x*100:.1f}%"
            ),
            title=f"{selected_model} 예측 결과"
        )

        fig.update_layout(
            yaxis_range=[0, 1],
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.info(
            "예측 확률은 선택한 모델이 계산한 "
            "분류 점수입니다."
        )

# =====================================
# TAB 2. 알고리즘 비교
# =====================================

with tab2:

    st.header("🏆 AI 알고리즘 성능 비교")

    st.write(
        "동일한 테스트 데이터를 사용하여 "
        "3가지 알고리즘의 성능을 비교합니다."
    )

    st.subheader("성능 평가표")

    display_df = result_df.copy()

    for col in ["정확도", "정밀도", "재현율", "F1 점수"]:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x*100:.2f}%"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("정확도 비교")

    fig = px.bar(
        result_df,
        x="알고리즘",
        y="정확도",
        color="알고리즘",
        text=result_df["정확도"].apply(
            lambda x: f"{x*100:.1f}%"
        ),
        title="3가지 알고리즘 정확도"
    )

    fig.update_layout(yaxis_range=[0, 1])

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    best = result_df.loc[
        result_df["정확도"].idxmax()
    ]

    st.success(
        f"🥇 가장 높은 정확도: {best['알고리즘']} "
        f"({best['정확도']*100:.2f}%)"
    )

    st.subheader("종합 성능 비교")

    metric_df = result_df.melt(
        id_vars="알고리즘",
        var_name="평가 항목",
        value_name="점수"
    )

    fig = px.bar(
        metric_df,
        x="알고리즘",
        y="점수",
        color="평가 항목",
        barmode="group",
        title="정확도, 정밀도, 재현율, F1 점수 비교"
    )

    fig.update_layout(yaxis_range=[0, 1])

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("혼동 행렬")

    chosen = st.selectbox(
        "확인할 알고리즘",
        list(trained_models.keys())
    )

    predictions = trained_models[chosen].predict(X_test)

    labels = trained_models[chosen].named_steps[
        "model"
    ].classes_

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={
            "x": "예측 종류",
            "y": "실제 종류",
            "color": "개체 수"
        },
        title=f"{chosen} 혼동 행렬"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# TAB 3. 시각화
# =====================================

with tab3:

    st.header("📈 펭귄 데이터 시각화")

    c1, c2 = st.columns(2)

    with c1:
        x_axis = st.selectbox(
            "가로축",
            features,
            index=0
        )

    with c2:
        y_axis = st.selectbox(
            "세로축",
            features,
            index=2
        )

    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color="species",
        title="펭귄 종류별 특징 산점도",
        hover_data=features
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("펭귄 종류별 평균 비교")

    avg_df = df.groupby(
        "species"
    )[features].mean().reset_index()

    selected_feature = st.selectbox(
        "비교할 신체 특징",
        features,
        key="average"
    )

    fig = px.bar(
        avg_df,
        x="species",
        y=selected_feature,
        color="species",
        text_auto=".2f",
        title=f"종류별 {selected_feature} 평균"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# TAB 4. 데이터 분석
# =====================================

with tab4:

    st.header("🔎 데이터 분석 및 결측치 확인")

    st.subheader("원본 데이터")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("결측치 현황")

    missing = df[features].isnull().sum()

    missing_df = pd.DataFrame({
        "항목": missing.index,
        "결측치 개수": missing.values
    })

    fig = px.bar(
        missing_df,
        x="항목",
        y="결측치 개수",
        color="항목",
        title="항목별 결측치 개수"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("기초 통계")

    st.dataframe(
        df[features].describe().round(2),
        use_container_width=True
    )

    st.subheader("펭귄 종류별 데이터 개수")

    count_df = df["species"].value_counts().reset_index()
    count_df.columns = ["펭귄 종류", "개체 수"]

    fig = px.pie(
        count_df,
        names="펭귄 종류",
        values="개체 수",
        title="펭귄 종류별 비율",
        hole=0.4
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("분석 데이터 다운로드")

    csv = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 분석 데이터 CSV 다운로드",
        data=csv,
        file_name="penguin_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

# =====================================
# 11. 알고리즘 설명
# =====================================

st.divider()

st.header("🧠 사용된 인공지능 알고리즘")

with st.expander("1. KNN 알고리즘"):
    st.write("""
    새로운 데이터와 가까운 기존 데이터 K개를 찾고,
    가장 많이 나타나는 종류로 분류하는 방법입니다.
    """)

with st.expander("2. 의사결정나무"):
    st.write("""
    여러 가지 조건을 기준으로 데이터를 나누면서
    펭귄의 종류를 판단하는 방법입니다.
    """)

with st.expander("3. 랜덤 포레스트"):
    st.write("""
    여러 개의 의사결정나무를 만들고,
    각각의 예측 결과를 종합하여 최종 종류를 결정하는 방법입니다.
    """)

st.divider()

st.caption(
    "🐧 Penguin AI Lab | Palmer Penguins CSV 기반 인공지능 수행평가"
)

