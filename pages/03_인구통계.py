
import re
from pathlib import Path
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# 페이지 설정
st.set_page_config(
    page_title="서울시 연령별 인구 분석",
    page_icon="📊",
    layout="wide"
)


# CSV 파일 읽기
# pages 폴더의 상위 폴더에서 CSV 파일을 찾음
@st.cache_data
def load_population_data():

    csv_path = Path(__file__).resolve().parent.parent / "population.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            "상위 폴더에서 population.csv 파일을 찾을 수 없습니다."
        )

    file_bytes = csv_path.read_bytes()

    for encoding in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            df = pd.read_csv(
                BytesIO(file_bytes),
                encoding=encoding
            )
            return df, encoding

        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일의 인코딩을 확인해주세요.")


# 한글 폰트 설정
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Malgun Gothic",
                 "Apple SD Gothic Neo",
                 "Noto Sans KR",
                 sans-serif;
}
</style>
""", unsafe_allow_html=True)


# 제목
st.title("📊 서울시 연령별 인구 분석")
st.write("행정구역, 성별, 연도를 선택하여 연령별 인구를 분석해보세요!")


# 데이터 불러오기
try:
    df, used_encoding = load_population_data()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# 행정구역 확인
if "행정구역" not in df.columns:
    st.error("CSV에서 행정구역 열을 찾을 수 없습니다.")
    st.stop()


# 행정구역명 정리
df["행정구역명"] = (
    df["행정구역"]
    .astype(str)
    .str.replace(r"\s*\([^)]*\)", "", regex=True)
    .str.strip()
)


# 연령별 열 찾기
age_pattern = re.compile(
    r"^(?P<year>\d{4})년_(?P<gender>계|남|여)_(?P<age>\d+~\d+세|100세 이상)$"
)

age_columns = []

for column in df.columns:

    match = age_pattern.match(str(column))

    if match:
        age_columns.append({
            "column": column,
            "year": int(match.group("year")),
            "gender": match.group("gender"),
            "age": match.group("age")
        })


if not age_columns:
    st.error("연령별 인구 데이터를 찾을 수 없습니다.")
    st.stop()


# 선택 가능한 연도
years = sorted(set(
    item["year"] for item in age_columns
))


# 성별
gender_map = {
    "전체": "계",
    "남성": "남",
    "여성": "여"
}


# 연령 순서
age_order = [
    "0~9세",
    "10~19세",
    "20~29세",
    "30~39세",
    "40~49세",
    "50~59세",
    "60~69세",
    "70~79세",
    "80~89세",
    "90~99세",
    "100세 이상"
]


# 사이드바 선택 메뉴
st.sidebar.header("🔎 분석 조건 선택")


district_options = (
    df["행정구역명"]
    .dropna()
    .drop_duplicates()
    .tolist()
)

district = st.sidebar.selectbox(
    "행정구역 선택",
    district_options
)

gender_label = st.sidebar.selectbox(
    "성별 선택",
    list(gender_map.keys())
)

year = st.sidebar.selectbox(
    "연도 선택",
    years,
    index=len(years) - 1
)


# 선택한 데이터 가져오기
selected_gender = gender_map[gender_label]

selected_row = df[
    df["행정구역명"] == district
].iloc[0]


selected_columns = [
    item for item in age_columns
    if item["year"] == year
    and item["gender"] == selected_gender
]


# 연령 순서 정렬
selected_columns.sort(
    key=lambda item: age_order.index(item["age"])
    if item["age"] in age_order else 999
)


# 그래프 데이터 생성
chart_data = []

for item in selected_columns:

    raw_value = selected_row[item["column"]]

    numeric_value = pd.to_numeric(
        str(raw_value).replace(",", "").strip(),
        errors="coerce"
    )

    chart_data.append({
        "연령대": item["age"],
        "인구수": 0 if pd.isna(numeric_value)
        else int(numeric_value)
    })


plot_df = pd.DataFrame(chart_data)


if plot_df.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()


# 주요 통계
total_population = int(plot_df["인구수"].sum())

peak = plot_df.loc[
    plot_df["인구수"].idxmax()
]


col1, col2, col3 = st.columns(3)

col1.metric(
    "선택 지역",
    district
)

col2.metric(
    "연령별 인구 합계",
    f"{total_population:,}명"
)

col3.metric(
    "인구가 가장 많은 연령대",
    peak["연령대"],
    f'{int(peak["인구수"]):,}명'
)


# 꺾은선 그래프
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=plot_df["연령대"],
        y=plot_df["인구수"],
        mode="lines+markers",
        name="인구수",

        line=dict(
            color="#FF1493",
            width=4
        ),

        marker=dict(
            color="#FF1493",
            size=9
        ),

        hovertemplate=
        "%{x}<br>인구수: %{y:,.0f}명<extra></extra>"
    )
)


# 그래프 기본 설정
fig.update_layout(
    title=f"{district} {year}년 {gender_label} 연령대별 인구",

    xaxis_title="나이",
    yaxis_title="인구수 (명)",

    template="plotly_white",

    font=dict(
        family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif",
        size=13,
        color="#222222"
    ),

    height=560,

    hovermode="x unified",

    margin=dict(
        l=30,
        r=30,
        t=75,
        b=35
    )
)


# 가로축: 10살 단위
fig.update_xaxes(
    type="category",

    categoryorder="array",
    categoryarray=age_order,

    tickmode="array",
    tickvals=age_order,

    showgrid=True,
    gridcolor="#E8E8E8",

    title_font=dict(size=15)
)


# 세로축: 1,000만 명 단위
y_max = max(
    10_000_000,
    (
        int(plot_df["인구수"].max())
        // 10_000_000 + 1
    ) * 10_000_000
)

fig.update_yaxes(
    range=[0, y_max],

    tickmode="linear",

    dtick=10_000_000,

    tickformat=",",

    showgrid=True,
    gridcolor="#DADADA",

    zeroline=True,
    zerolinecolor="#999999",

    title_font=dict(size=15)
)


# Streamlit 그래프 출력
st.plotly_chart(
    fig,
    use_container_width=True
)


# 데이터 표
st.subheader("📋 연령대별 인구 데이터")

display_df = plot_df.copy()

display_df["인구수"] = display_df["인구수"].map(
    lambda value: f"{value:,}명"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# 인코딩 정보
st.caption(
    f"CSV 인코딩: {used_encoding}"
)

st.info(
    "세로축 눈금은 1,000만 명 단위입니다. "
    "자치구별 인구수는 1,000만 명보다 적으므로 "
    "그래프가 아래쪽에 작게 표시될 수 있습니다."
)
