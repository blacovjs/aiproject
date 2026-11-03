
import streamlit as st
import pandas as pd
from pathlib import Path
import html

st.set_page_config(
    page_title="국가별 MBTI 비율",
    page_icon="🌍",
    layout="wide"
)

# CSV 파일 경로
CSV_PATH = Path(__file__).parent / "countriesMBTI_16types.csv"


@st.cache_data
def load_data():
    data = pd.read_csv(CSV_PATH)

    if "Country" not in data.columns:
        raise ValueError("CSV 파일에 Country 열이 없습니다.")

    mbti_columns = [
        column for column in data.columns
        if column != "Country"
    ]

    data[mbti_columns] = data[mbti_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    data = data.dropna(subset=["Country"])
    data[mbti_columns] = data[mbti_columns].fillna(0)

    return data, mbti_columns


def make_blue_gradient(count):
    # 1위는 빨간색
    colors = ["#E53935"]

    if count <= 1:
        return colors

    # 나머지는 파란색에서 연한 파란색으로
    dark_blue = (21, 101, 192)
    light_blue = (187, 222, 251)

    for i in range(count - 1):
        t = i / max(count - 2, 1)

        r = round(
            dark_blue[0]
            + (light_blue[0] - dark_blue[0]) * t
        )

        g = round(
            dark_blue[1]
            + (light_blue[1] - dark_blue[1]) * t
        )

        b = round(
            dark_blue[2]
            + (light_blue[2] - dark_blue[2]) * t
        )

        colors.append(f"rgb({r}, {g}, {b})")

    return colors


st.title("🌍 국가별 MBTI 비율")

st.caption(
    "국가를 선택하면 해당 국가의 MBTI 16가지 유형별 비율을 확인할 수 있어요."
)

# 데이터 불러오기
try:
    df, mbti_columns = load_data()

except FileNotFoundError:
    st.error(
        "CSV 파일을 찾을 수 없습니다. "
        "countriesMBTI_16types.csv 파일을 "
        "app.py와 같은 폴더에 넣어주세요."
    )
    st.stop()

except Exception as error:
    st.error(f"데이터 오류: {error}")
    st.stop()


# 국가 선택
countries = sorted(
    df["Country"].astype(str).unique().tolist()
)

default_country = (
    "South Korea"
    if "South Korea" in countries
    else countries[0]
)

selected_country = st.selectbox(
    "🌎 국가 선택",
    countries,
    index=countries.index(default_country)
)


# 선택한 국가 데이터
country_data = df[
    df["Country"].astype(str) == selected_country
].iloc[0]


# MBTI 비율 계산 및 정렬
chart_data = pd.DataFrame({
    "MBTI": mbti_columns,
    "비율": [
        float(country_data[mbti]) * 100
        for mbti in mbti_columns
    ]
})

chart_data = chart_data.sort_values(
    by="비율",
    ascending=False
).reset_index(drop=True)


# 그래프 색상
colors = make_blue_gradient(len(chart_data))


# 그래프 제목
st.subheader(f"📊 {selected_country}의 MBTI 유형별 비율")


# 최대 비율
max_value = max(chart_data["비율"].max(), 1)


# 그래프 HTML
chart_html = """
<div style="
    width: 100%;
    padding: 15px 5px;
    font-family: Arial, sans-serif;
">
"""

for index, row in chart_data.iterrows():

    mbti = html.escape(str(row["MBTI"]))
    value = float(row["비율"])
    color = colors[index]

    # 막대 길이
    width = (value / max_value) * 100

    chart_html += f"""
    <div style="
        display: flex;
        align-items: center;
        margin-bottom: 13px;
        gap: 10px;
    ">

        <div style="
            width: 55px;
            min-width: 55px;
            font-weight: bold;
            font-size: 14px;
            color: #333;
        ">
            {mbti}
        </div>

        <div style="
            flex-grow: 1;
            background-color: #f1f3f5;
            border-radius: 6px;
            height: 28px;
            overflow: visible;
        ">

            <div style="
                width: {width:.2f}%;
                background-color: {color};
                height: 28px;
                border-radius: 6px;
                min-width: 2px;
            ">
            </div>

        </div>

        <div style="
            width: 65px;
            min-width: 65px;
            font-size: 13px;
            text-align: right;
            color: #333;
        ">
            {value:.2f}%
        </div>

    </div>
    """

chart_html += "</div>"


# 그래프 표시
st.markdown(
    chart_html,
    unsafe_allow_html=True
)


# 가장 높은 비율의 MBTI
top_type = chart_data.iloc[0]

st.success(
    f"📌 {selected_country}에서 가장 높은 비율을 보이는 유형은 "
    f"{top_type['MBTI']} ({top_type['비율']:.2f}%) 입니다."
)


# 데이터 표
with st.expander("📋 비율 데이터 표 보기"):

    display_data = chart_data.copy()

    display_data["비율"] = display_data["비율"].map(
        lambda value: f"{value:.2f}%"
    )

    st.dataframe(
        display_data,
        hide_index=True,
        use_container_width=True
    )


st.caption(
    "※ 이 데이터는 국가별 MBTI 유형 비율을 나타내는 참고용 통계입니다."
)
