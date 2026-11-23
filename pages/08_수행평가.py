
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="Penguin AI World 🐧",
    page_icon="🐧",
    layout="wide"
)

# ==========================================
# 2. 귀여운 디자인 CSS
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Jua&display=swap');

html, body, [class*="css"] {
    font-family: 'Jua', sans-serif;
}

.stApp {
    background: linear-gradient(
        180deg,
        #e8f6ff 0%,
        #f5fbff 50%,
        #ffffff 100%
    );
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(135deg, #8ed8f8, #c5edff);
    padding: 35px 20px;
    border-radius: 28px;
    text-align: center;
    color: #174568;
    box-shadow: 0 8px 20px rgba(67, 151, 190, 0.15);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 43px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 19px;
}

.penguin-card {
    background: white;
    padding: 22px;
    border-radius: 22px;
    border: 2px solid #d8effb;
    box-shadow: 0 5px 15px rgba(73, 143, 176, 0.10);
    margin-bottom: 15px;
}

.section-title {
    color: #245b7a;
    font-size: 25px;
    margin-top: 20px;
    margin-bottom: 15px;
}

.info-box {
    background: #ffffff;
    border-left: 6px solid #8ed8f8;
    padding: 18px;
    border-radius: 15px;
    margin-bottom: 15px;
    color: #31546a;
}

div[data-testid="stMetric"] {
    background: white;
    border: 2px solid #d8effb;
    padding: 17px;
    border-radius: 18px;
    box-shadow: 0 4px 12px rgba(73, 143, 176, 0.08);
}

.stButton > button {
    background: linear-gradient(90deg, #69c5ed, #91dcf7);
    color: #174568;
    border: none;
    border-radius: 15px;
    padding: 12px 25px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background: #4db5e5;
    color: white;
}

.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 12px;
    padding: 12px 18px;
    margin-right: 5px;
}

.stTabs [aria-selected="true"] {
    background: #bdeaff;
    color: #174568;
}

.stSelectbox, .stNumberInput {
    background: white;
    border-radius: 12px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 상단 화면
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🐧 Penguin AI World ❄️</h1>
    <p>귀여운 펭귄 친구들의 종을 AI로 알아보자!</p>
    <p>🔍 펭귄 연구소에 온 걸 환영해요!</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("penguins.csv")

try:
    df = load_data()
except Exception:
    st.error("🐧 penguins.csv 파일을 찾을 수 없어요!")
    st.info("app.py와 penguins.csv 파일이 같은 폴더에 있는지 확인해 주세요.")
    st.stop()

# 불필요한 인덱스 열 제거
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

# 컬럼명 정리
df.columns = df.columns.str.strip()

# 필요한 컬럼 확인
if "species" not in df.columns:
    st.error("데이터에 species 열이 없습니다.")
    st.stop()

# 타깃이 없는 행 제거
df = df.dropna(subset=["species"]).copy()

# 분석에 사용할 특성
features = [
    "island",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "sex"
]

features = [col for col in features if col in df.columns]

if not features:
    st.error("분석에 사용할 특성을 찾을 수 없습니다.")
    st.stop()

X = df[features].copy()
y = df["species"].astype(str)

# 숫자형과 범주형 컬럼 구분
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [
    col for col in features if col not in numeric_features
]

# ==========================================
# 5. 데이터 전처리
# ==========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================================
# 6. AI 모델 준비
# ==========================================

models = {
    "KNN 🐧": KNeighborsClassifier(n_neighbors=5),
    "의사결정나무 🌳": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "랜덤포레스트 🌲": RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )
}

# ==========================================
# 7. 학습 및 평가
# ==========================================

@st.cache_resource
def train_models(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    trained_models = {}
    results = []

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        trained_models[name] = pipeline

        results.append({
            "모델": name,
            "정확도": accuracy_score(y_test, predictions),
            "정밀도": precision_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            ),
            "재현율": recall_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            ),
            "F1 점수": f1_score(
                y_test, predictions,
                average="weighted",
                zero_division=0
            )
        })

    return (
        trained_models,
        pd.DataFrame(results),
        X_test,
        y_test
    )

try:
    trained_models, results_df, X_test, y_test = train_models(
        X, y, features
    )
except Exception as e:
    st.error("AI 모델을 학습하는 중 문제가 발생했어요.")
    st.exception(e)
    st.stop()

# ==========================================
# 8. 메뉴
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🐧 펭귄 예측",
    "🤖 AI 모델 비교",
    "📊 데이터 분석",
    "❄️ 펭귄 도감"
])

# ==========================================
# TAB 1. 펭귄 예측
# ==========================================

with tab1:

    st.markdown(
        '<div class="section-title">🐧 어떤 펭귄일까요?</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    펭귄의 특징을 입력하면 AI가 어떤 종인지 예측해 줘요!
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📏 몸의 특징")

        input_values = {}

        for col in numeric_features:
            valid = pd.to_numeric(df[col], errors="coerce").dropna()

            if len(valid) == 0:
                continue

            minimum = float(valid.min())
            maximum = float(valid.max())
            median = float(valid.median())

            input_values[col] = st.number_input(
                col,
                min_value=minimum,
                max_value=maximum,
                value=median,
                step=0.1,
                format="%.2f"
            )

    with col2:
        st.markdown("### 🏝️ 추가 정보")

        for col in categorical_features:
            options = sorted(
                df[col].dropna().astype(str).unique().tolist()
            )

            if options:
                input_values[col] = st.selectbox(
                    col,
                    options
                )

    selected_model = st.selectbox(
        "🤖 사용할 AI 모델을 선택하세요",
        list(trained_models.keys())
    )

    if st.button("🐧 내 펭귄의 종 알아보기", use_container_width=True):

        input_df = pd.DataFrame(
            [input_values],
            columns=features
        )

        model = trained_models[selected_model]

        prediction = model.predict(input_df)[0]

        probabilities = model.predict_proba(input_df)[0]
        classes = model.named_steps["classifier"].classes_

        st.markdown("---")

        st.markdown(
            '<div class="section-title">🎉 AI 예측 결과!</div>',
            unsafe_allow_html=True
        )

        st.success(f"🐧 이 펭귄은 **{prediction}** 종으로 예측되었어요!")

        probability_df = pd.DataFrame({
            "펭귄 종": classes,
            "예측 확률": probabilities
        })

        fig = px.bar(
            probability_df,
            x="펭귄 종",
            y="예측 확률",
            color="펭귄 종",
            text_auto=".1%",
            title="🐧 펭귄 종별 예측 확률"
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis_tickformat=".0%",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        confidence = float(max(probabilities))

        st.metric(
            "AI의 가장 높은 예측 확률",
            f"{confidence:.1%}"
        )

        if confidence >= 0.8:
            st.balloons()
            st.info("정말 높은 확률로 예측했어요! 🎊")
        else:
            st.info("다른 펭귄 종일 가능성도 있으니 참고해 주세요! ❄️")

# ==========================================
# TAB 2. AI 모델 비교
# ==========================================

with tab2:

    st.markdown(
        '<div class="section-title">🤖 AI 모델 대결!</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    세 가지 AI 모델 중 어떤 모델이 펭귄 종을 가장 잘 맞힐까요?
    </div>
    """, unsafe_allow_html=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    best_model = results_df.loc[
        results_df["정확도"].idxmax()
    ]

    with metric_col1:
        st.metric(
            "🏆 가장 높은 정확도",
            f"{best_model['정확도']:.1%}"
        )

    with metric_col2:
        st.metric(
            "🤖 최고 모델",
            best_model["모델"]
        )

    with metric_col3:
        st.metric(
            "📚 학습 데이터",
            f"{len(df)}개"
        )

    st.markdown("### 📊 모델별 성능 비교")

    metric_choice = st.selectbox(
        "비교할 평가 지표",
        ["정확도", "정밀도", "재현율", "F1 점수"]
    )

    fig = px.bar(
        results_df,
        x="모델",
        y=metric_choice,
        color="모델",
        text=results_df[metric_choice].apply(
            lambda x: f"{x:.1%}"
        ),
        title=f"🐧 AI 모델별 {metric_choice} 비교"
    )

    fig.update_layout(
        yaxis=dict(range=[0, 1.05]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 전체 평가 결과")

    display_results = results_df.copy()

    for col in ["정확도", "정밀도", "재현율", "F1 점수"]:
        display_results[col] = display_results[col].map(
            lambda x: f"{x:.2%}"
        )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    # 혼동 행렬
    st.markdown("### 🔍 모델별 예측 결과 분석")

    selected_matrix_model = st.selectbox(
        "혼동 행렬을 확인할 모델",
        list(trained_models.keys()),
        key="matrix_model"
    )

    matrix_model = trained_models[selected_matrix_model]
    matrix_predictions = matrix_model.predict(X_test)

    labels = sorted(y.unique())

    cm = confusion_matrix(
        y_test,
        matrix_predictions,
        labels=labels
    )

    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={
            "x": "AI 예측 종",
            "y": "실제 종",
            "color": "개수"
        },
        title="🐧 실제 펭귄 종과 AI 예측 비교"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "가로축은 AI가 예측한 종이고 세로축은 실제 종이에요."
    )

# ==========================================
# TAB 3. 데이터 분석
# ==========================================

with tab3:

    st.markdown(
        '<div class="section-title">📊 펭귄 데이터 연구소</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    펭귄들의 몸의 특징과 종별 차이를 그래프로 살펴봐요!
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("🐧 전체 펭귄 수", f"{len(df)}마리")

    with c2:
        st.metric("❄️ 펭귄 종류", f"{y.nunique()}종")

    with c3:
        st.metric("📋 데이터 특성", f"{len(features)}개")

    # 종별 개수
    st.markdown("### 🐧 펭귄 종별 개수")

    species_count = (
        df["species"]
        .value_counts()
        .rename_axis("펭귄 종")
        .reset_index(name="개체 수")
    )

    fig = px.pie(
        species_count,
        names="펭귄 종",
        values="개체 수",
        hole=0.45,
        title="펭귄 종별 비율",
        color_discrete_sequence=[
            "#75C8EA",
            "#A5DDF5",
            "#D1EDFA"
        ]
    )

    st.plotly_chart(fig, use_container_width=True)

    # 결측치
    st.markdown("### 🧩 결측치 확인")

    missing = df.isnull().sum().reset_index()
    missing.columns = ["컬럼", "결측치 개수"]

    fig = px.bar(
        missing,
        x="컬럼",
        y="결측치 개수",
        color="결측치 개수",
        title="컬럼별 결측치 개수",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 특성 선택
    available_numeric = [
        col for col in numeric_features
        if df[col].notna().any()
    ]

    if len(available_numeric) >= 2:

        st.markdown("### 📈 펭귄 특징 비교")

        x_axis = st.selectbox(
            "가로축",
            available_numeric,
            index=0
        )

        y_axis = st.selectbox(
            "세로축",
            available_numeric,
            index=1 if len(available_numeric) > 1 else 0
        )

        plot_df = df.dropna(
            subset=[x_axis, y_axis, "species"]
        )

        fig = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color="species",
            title="🐧 펭귄 종별 특징 분포",
            hover_data=features
        )

        fig.update_traces(marker=dict(size=11, opacity=0.8))

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # 데이터 미리보기
    st.markdown("### 📚 펭귄 데이터 구경하기")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    csv = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "📥 펭귄 데이터 다운로드",
        data=csv,
        file_name="penguins_data.csv",
        mime="text/csv"
    )

# ==========================================
# TAB 4. 펭귄 도감
# ==========================================

with tab4:

    st.markdown(
        '<div class="section-title">📖 귀여운 펭귄 도감</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
    우리 AI 연구소에서 알아볼 수 있는 펭귄 친구들이에요!
    </div>
    """, unsafe_allow_html=True)

    species_list = sorted(df["species"].unique())

    penguin_images = {
        "Adelie": "cute adelie penguin standing on snow",
        "Chinstrap": "cute chinstrap penguin standing on snow",
        "Gentoo": "cute gentoo penguin standing on snow"
    }

    penguin_descriptions = {
        "Adelie": "작은 몸과 짧은 부리가 특징인 펭귄 친구예요!",
        "Chinstrap": "턱 아래 검은 줄무늬가 특징인 펭귄 친구예요!",
        "Gentoo": "머리 위의 흰색 무늬와 주황색 부리가 특징이에요!"
    }

    cols = st.columns(3)

    for i, species in enumerate(species_list):

        with cols[i % 3]:

            st.markdown(
                '<div class="penguin-card">',
                unsafe_allow_html=True
            )

            if species in penguin_images:
                st.image(
                    "https://images.unsplash.com/photo-1551986782-d0169b3f8fa7?w=600",
                    use_container_width=True
                )

            st.subheader(f"🐧 {species}")

            description = penguin_descriptions.get(
                species,
                "남극의 귀여운 펭귄 친구예요!"
            )

            st.write(description)

            count = int((df["species"] == species).sum())

            st.metric("데이터 속 개체 수", f"{count}마리")

            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 하단
# ==========================================

st.markdown("---")

st.markdown("""
<div style="
    text-align:center;
    background:#dff3ff;
    padding:25px;
    border-radius:20px;
    color:#315c75;
">
    <h3>🐧 Penguin AI World ❄️</h3>
    <p>AI와 함께 펭귄 친구들을 알아가는 즐거운 연구소!</p>
    <p>Made with Streamlit 💙</p>
</div>
""", unsafe_allow_html=True)
