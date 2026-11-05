import re
from pathlib import Path
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="서울시 연령별 인구 분석",
    page_icon="📊",
    layout="wide"
)


# CSV 파일 불러오기
# pages 폴더의 한 단계 위에서 population.csv를 찾습니다.
@st.cache_data
def load_population_data():
    csv_path = Path(__file__).resolve().parent.parent / "population.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {csv_path}"
        )

    file_bytes = csv_path.read_bytes()
    last_error = None

    # 한글 CSV 인코딩을 순서대로 확인합니다.
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            data = pd.read_csv(
                BytesIO(file_bytes),
                encoding=encoding
            )
            return data, encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeError(
        "CSV 인코딩을 확인할 수 없습니다. "
        "UTF-8-SIG 또는 CP949로 저장해주세요."
    ) from last_error


# 한글 폰트 설정
st.markdown(
    """
    <style>
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: "Malgun Gothic", "Apple SD Gothic Neo",
                     "Noto Sans KR", sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📊 서울시 연령별 인구 분석")
st.caption(
    "행정구역·성별·연도를 선택해 연령별 인구를 확인하고, "
    "연령대별 인구가 많은 자치구를 비교합니다."
)


# 데이터 불러오기
try:
    df, used_encoding = load_population_data()
except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다: {error}")
    st.stop()


if "행정구역" not in df.columns:
    st.error("CSV 파일에 '행정구역' 열이 없습니다.")
    st.stop()


# 행정구역명에서 괄호 안 행정코드 제거
df["행정구역명"] = (
    df["행정구역"].astype(str)
    .str.replace(r"\s*\([^)]*\)", "", regex=True)
    .str.strip()
)


# 연령별 인구 열 분석
# 예: 2025년_남_0~9세
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
    st.error(
        "연령별 인구 열을 찾지 못했습니다. "
        "CSV의 열 이름을 확인해주세요."
    )
    st.stop()


years = sorted({
    item["year"] for item in age_columns
})

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

gender_map = {
    "전체": "계",
    "남성": "남",
    "여성": "여"
}


# 인구수 문자열을 숫자로 변환
def to_number(value):
    return pd.to_numeric(
        str(value).replace(",", "").strip(),
        errors="coerce"
    )


# ==================================================
# 1. 행정구역별 연령대 인구 꺾은선 그래프
# ==================================================

st.sidebar.header("🔎 행정구역별 분석")


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


selected_gender = gender_map[gender_label]

selected_row = df.loc[
    df["행정구역명"] == district
].iloc[0]


selected_columns = [
    item for item in age_columns
    if item["year"] == year
    and item["gender"] == selected_gender
]


selected_columns.sort(
    key=lambda item: age_order.index(item["age"])
    if item["age"] in age_order else 999
)


line_data = []

for item in selected_columns:
    value = to_number(
        selected_row[item["column"]]
    )

    line_data.append({
        "연령대": item["age"],
        "인구수": 0 if pd.isna(value) else int(value)
    })


line_df = pd.DataFrame(line_data)


st.subheader("1. 행정구역별 연령대 인구")


if line_df.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")

else:
    total_population = int(
        line_df["인구수"].sum()
    )

    peak = line_df.loc[
        line_df["인구수"].idxmax()
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
    line_fig = go.Figure()

    line_fig.add_trace(
        go.Scatter(
            x=line_df["연령대"],
            y=line_df["인구수"],
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
            hovertemplate=(
                "%{x}<br>"
                "인구수: %{y:,.0f}명"
                "<extra></extra>"
            )
        )
    )


    line_fig.update_layout(
        title=f"{district} {year}년 {gender_label} 연령대별 인구",
        xaxis_title="나이",
        yaxis_title="인구수 (명)",
        template="plotly_white",
        font=dict(
            family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif",
            size=13,
            color="#222222"
        ),
        height=540,
        hovermode="x unified",
        margin=dict(
            l=30,
            r=30,
            t=75,
            b=35
        )
    )


    line_fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=age_order,
        tickmode="array",
        tickvals=age_order,
        showgrid=True,
        gridcolor="#E8E8E8"
    )


    # 세로축 눈금 간격 1,000만 명
    y_max = max(
        10_000_000,
        (
            int(line_df["인구수"].max()) // 10_000_000 + 1
        ) * 10_000_000
    )


    line_fig.update_yaxes(
        range=[0, y_max],
        tickmode="linear",
        dtick=10_000_000,
        tickformat=",",
        showgrid=True,
        gridcolor="#DADADA"
    )


    st.plotly_chart(
        line_fig,
        use_container_width=True
    )


    # 인구수 표
    display_line_df = line_df.copy()

    display_line_df["인구수"] = (
        display_line_df["인구수"]
        .map(lambda value: f"{value:,}명")
    )

    st.dataframe(
        display_line_df,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# ==================================================
# 2. 연령대별 자치구 인구 순위 막대그래프
# ==================================================

st.subheader("2. 연령대별 인구가 많은 자치구")

st.write(
    "연령대와 연도를 선택하면 해당 연령대의 남녀 인구를 "
    "자치구별로 비교합니다. "
    "남성은 핑크색, 여성은 초록색입니다."
)


control1, control2 = st.columns(2)


with control1:
    rank_age = st.selectbox(
        "비교할 연령대 선택",
        age_order,
        index=2,
        key="rank_age"
    )


with control2:
    rank_year = st.selectbox(
        "비교할 연도 선택",
        years,
        index=len(years) - 1,
        key="rank_year"
    )


# 서울특별시 전체 행은 제외하고 25개 자치구만 비교
district_df = df[
    df["행정구역명"] != "서울특별시"
].copy()


# 남성 인구 열 찾기
male_column = next(
    (
        item["column"]
        for item in age_columns
        if item["year"] == rank_year
        and item["gender"] == "남"
        and item["age"] == rank_age
    ),
    None
)


# 여성 인구 열 찾기
female_column = next(
    (
        item["column"]
        for item in age_columns
        if item["year"] == rank_year
        and item["gender"] == "여"
        and item["age"] == rank_age
    ),
    None
)


if male_column is None or female_column is None:

    st.error(
        "선택한 연도와 연령대의 남녀 데이터를 찾을 수 없습니다."
    )

else:

    rank_df = pd.DataFrame({
        "행정구역": district_df["행정구역명"],
        "남성": district_df[male_column].apply(to_number),
        "여성": district_df[female_column].apply(to_number)
    })


    rank_df = rank_df.fillna(0)

    rank_df["남성"] = rank_df["남성"].astype(int)
    rank_df["여성"] = rank_df["여성"].astype(int)

    rank_df["합계"] = (
        rank_df["남성"] + rank_df["여성"]
    )


    # 남녀 합계가 많은 순서로 정렬
    rank_df = rank_df.sort_values(
        "합계",
        ascending=False
    ).reset_index(drop=True)


    if not rank_df.empty:

        top = rank_df.iloc[0]

        st.success(
            f"🏆 {rank_year}년 {rank_age} 인구가 가장 많은 자치구는 "
            f"**{top['행정구역']}**입니다. "
            f"(남녀 합계 {int(top['합계']):,}명)"
        )


        # 남녀 인구 막대그래프
        bar_fig = go.Figure()


        # 남성: 핑크색
        bar_fig.add_trace(
            go.Bar(
                x=rank_df["행정구역"],
                y=rank_df["남성"],
                name="남성",
                marker_color="#FF69B4",
                hovertemplate=(
                    "%{x}<br>"
                    "남성: %{y:,.0f}명"
                    "<extra></extra>"
                )
            )
        )


        # 여성: 초록색
        bar_fig.add_trace(
            go.Bar(
                x=rank_df["행정구역"],
                y=rank_df["여성"],
                name="여성",
                marker_color="#2EAD65",
                hovertemplate=(
                    "%{x}<br>"
                    "여성: %{y:,.0f}명"
                    "<extra></extra>"
                )
            )
        )


        bar_fig.update_layout(
            title=f"{rank_year}년 {rank_age} 자치구별 남녀 인구 비교",
            xaxis_title="행정구역",
            yaxis_title="인구수 (명)",
            barmode="group",
            template="plotly_white",
            font=dict(
                family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif",
                size=12,
                color="#222222"
            ),
            height=600,
            hovermode="x unified",
            legend_title="성별",
            margin=dict(
                l=30,
                r=30,
                t=75,
                b=100
            )
        )


        bar_fig.update_xaxes(
            tickangle=-45,
            showgrid=False
        )

        bar_fig.update_yaxes(
            tickformat=",",
            showgrid=True,
            gridcolor="#E5E5E5"
        )


        st.plotly_chart(
            bar_fig,
            use_container_width=True
        )


        # 자치구별 인구 순위 표
        st.subheader("📋 자치구별 남녀 인구 순위")

        display_rank_df = rank_df[
            ["행정구역", "남성", "여성", "합계"]
        ].copy()


        for column in ["남성", "여성", "합계"]:

            display_rank_df[column] = (
                display_rank_df[column]
                .map(lambda value: f"{value:,}명")
            )


        st.dataframe(
            display_rank_df,
            use_container_width=True,
            hide_index=True
        )


# 데이터 인코딩 표시
st.caption(f"데이터 인코딩: {used_encoding}")

st.caption(
    "출처: 업로드한 population.csv 파일"
)
