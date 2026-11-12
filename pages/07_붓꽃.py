
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

st.set_page_config(
    page_title="붓꽃 품종 예측",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 붓꽃 품종 예측 프로그램")
st.write("꽃잎의 길이와 너비를 입력하면 AI가 붓꽃 품종을 예측합니다.")

# 데이터 불러오기
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Iris.csv")
        df.columns = df.columns.str.strip()

        required = ["꽃잎길이", "꽃잎너비", "품종"]

        if all(col in df.columns for col in required):
            return df[required].dropna()

    except Exception:
        pass

    iris = load_iris(as_frame=True)
    df = iris.frame

    df = df.rename(columns={
        "petal length (cm)": "꽃잎길이",
        "petal width (cm)": "꽃잎너비"
    })

    df["품종"] = df["target"].map({
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    })

    return df[["꽃잎길이", "꽃잎너비", "품종"]]

df = load_data()

# 학습 데이터
X = df[["꽃잎길이", "꽃잎너비"]]
y = df["품종"]

# 랜덤 포레스트 모델
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

# 사용자 입력
st.subheader("1. 꽃잎 크기 입력")

col1, col2 = st.columns(2)

with col1:
    petal_length = st.number_input(
        "꽃잎 길이 (cm)",
        min_value=0.1,
        max_value=10.0,
        value=4.5,
        step=0.1
    )

with col2:
    petal_width = st.number_input(
        "꽃잎 너비 (cm)",
        min_value=0.1,
        max_value=5.0,
        value=1.5,
        step=0.1
    )

# 예측
input_data = pd.DataFrame({
    "꽃잎길이": [petal_length],
    "꽃잎너비": [petal_width]
})

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0]

species_name = {
    "Setosa": "세토사",
    "Versicolor": "버시컬러",
    "Virginica": "버지니카",
    "Iris-setosa": "세토사",
    "Iris-versicolor": "버시컬러",
    "Iris-virginica": "버지니카"
}

# 예측 결과
st.subheader("2. AI 품종 예측 결과")

st.success(
    f"🌸 예측된 품종: {species_name.get(prediction, prediction)}"
)

st.metric(
    "예측 확률",
    f"{max(probability) * 100:.1f}%"
)

# 품종별 예측 확률
prob_df = pd.DataFrame({
    "품종": [
        species_name.get(c, c)
        for c in model.classes_
    ],
    "예측 확률": probability * 100
})

st.write("### 품종별 예측 확률")
st.bar_chart(
    prob_df.set_index("품종")
)

# 그래프
st.subheader("3. 품종 분류 시각화")

st.write(
    "배경색은 AI가 예측하는 품종별 영역입니다. "
    "초록색 별은 사용자가 입력한 꽃잎 크기입니다."
)

x_min = min(X["꽃잎길이"].min(), petal_length) - 0.5
x_max = max(X["꽃잎길이"].max(), petal_length) + 0.5

y_min = min(X["꽃잎너비"].min(), petal_width) - 0.3
y_max = max(X["꽃잎너비"].max(), petal_width) + 0.3

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = pd.DataFrame({
    "꽃잎길이": xx.ravel(),
    "꽃잎너비": yy.ravel()
})

Z = model.predict(grid)

class_names = list(model.classes_)

Z = np.array([
    class_names.index(item)
    for item in Z
]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 6))

cmap = ListedColormap([
    "#F6D77A",
    "#8DD3C7",
    "#A6B9E8"
])

ax.contourf(
    xx,
    yy,
    Z,
    levels=np.arange(len(class_names) + 1) - 0.5,
    cmap=cmap,
    alpha=0.35
)

# 실제 데이터 표시
for species in class_names:

    subset = df[df["품종"] == species]

    ax.scatter(
        subset["꽃잎길이"],
        subset["꽃잎너비"],
        label=species_name.get(species, species),
        s=35,
        alpha=0.75
    )

# 사용자 입력값 초록색 별
ax.scatter(
    petal_length,
    petal_width,
    color="green",
    marker="*",
    s=450,
    edgecolors="black",
    linewidths=1.2,
    label="사용자 입력값",
    zorder=10
)

ax.set_xlabel("꽃잎 길이 (cm)")
ax.set_ylabel("꽃잎 너비 (cm)")
ax.set_title("붓꽃 품종 예측 분류 영역")

ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)

# 분류 이유
st.subheader("4. 품종 예측 이유")

st.write(
    f"""
    - 입력한 꽃잎 길이: {petal_length} cm
    - 입력한 꽃잎 너비: {petal_width} cm
    - 예측 품종: {species_name.get(prediction, prediction)}

    초록색 별이 위치한 배경색 영역이
    AI가 해당 품종으로 분류한 영역입니다.

    랜덤 포레스트는 여러 결정 트리의 예측 결과를
    종합하여 최종 품종을 결정합니다.
    """
)

# 특성 중요도
st.subheader("5. 분류에 영향을 준 특성")

importance_df = pd.DataFrame({
    "특성": ["꽃잎 길이", "꽃잎 너비"],
    "중요도": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "중요도",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("특성")
)

st.caption(
    "사용 알고리즘: Random Forest Classifier"
)
