
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
# 현재 파이썬 파일은 pages 폴더에 있고
# CSV 파일은 그 상위 폴더에 있음
CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "countriesMBTI_16types.csv"
)


@st.cache_data
def load_data():
    # CSV 파일 불러오기
    data = pd.read_csv(CSV_PATH)

    if "Country" not in data.columns:
        raise ValueError(
            "CSV 파일에 Country 열이 없습니다."
        )

    # 국가 열을 제외한 MBTI 유형 열
    mbti_columns = [
        column for column in data.columns
        if column != "Country"
    ]

    # 비율 데이터를 숫자로 변환
    data[mbti_columns] = data[mbti_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # 결측값 처리
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

        colors.append(
            f"rgb({r}, {g}, {b})"
        )

    return colors


# 페이지 제목
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
        "countriesMBTI_16types.csv 파일이 "
        "프로젝트 상위 폴더에 있는지 확인해주세요."
    )

    st.stop()

except Exception as error:

    st.error(
        f"데이터를 불러오는 중 오류가 발생했습니다: {error}"
    )

    st.stop()


# 국가 목록
countries = sorted(
    df["Country"].astype(str).unique().tolist()
)


# 기본 선택 국가
default_country = (
    "South Korea"
    if "South Korea" in countries
    else countries[0]
)


# 국가 선택
selected_country = st.selectbox(
    "🌎 국가 선택",
    countries,
    index=countries.index(default_country)
)


# 선택한 국가의 데이터
country_data = df[
    df["Country"].astype(str) == selected_country
].iloc[0]


# MBTI 비율 계산
chart_data = pd.DataFrame({
    "MBTI": mbti_columns,
    "비율": [
        float(country_data[mbti]) * 100
        for mbti in mbti_columns
    ]
})


# 비율이 높은 순서로 정렬
chart_data = chart_data.sort_values(
    by="비율",
    ascending=False
).reset_index(drop=True)


# 그래프 색상 생성
colors = make_blue_gradient(
    len(chart_data)
)


# 그래프 제목
st.subheader(
    f"📊 {selected_country}의 MBTI 유형별 비율"
)


# 가장 높은 비율
max_value = max(
    chart_data["비율"].max(),
    1
)


# HTML 그래프 시작
chart_html = """
<div style="
    width: 100%;
    padding: 15px 5px;
    font-family: Arial, sans-serif;
">
"""


# MBTI 막대그래프 생성
for index, row in chart_data.iterrows():

    mbti = html.escape(
        str(row["MBTI"])
    )

    value = float(
        row["비율"]
    )

    color = colors[index]

    # 막대 길이 계산
    width = (
        value / max_value
    ) * 100

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
            overflow: hidden;
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


# HTML 그래프 종료
chart_html += "</div>"


# 그래프 화면 출력
st.markdown(
    chart_html,
    unsafe_allow_html=True
)


# 가장 높은 비율의 MBTI 유형
top_type = chart_data.iloc[0]


# 결과 안내
st.success(
    f"📌 {selected_country}에서 가장 높은 비율을 보이는 유형은 "
    f"{top_type['MBTI']} ({top_type['비율']:.2f}%) 입니다."
)


# 데이터 표
with st.expander("📋 MBTI 비율 데이터 표 보기"):

    display_data = chart_data.copy()

    display_data["비율"] = display_data["비율"].map(
        lambda value: f"{value:.2f}%"
    )

    st.dataframe(
        display_data,
        hide_index=True,
        use_container_width=True
    )


# 안내 문구
st.caption(
    "※ 이 데이터는 국가별 MBTI 유형 비율을 나타내는 참고용 통계입니다."
)
