
import streamlit as st
import pandas as pd
from pathlib import Path
import html

st.set_page_config(
    page_title="국가별 MBTI 분석",
    page_icon="🌍",
    layout="wide"
)

# ==========================================
# CSV 파일 경로
# 파이썬 파일은 pages 폴더에 있음
# CSV 파일은 프로젝트 최상위 폴더에 있음
# ==========================================

CSV_PATH = Path(__file__).resolve().parent.parent / "countriesMBTI_16types.csv"


# ==========================================
# 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    data = pd.read_csv(CSV_PATH)

    mbti_columns = [
        col for col in data.columns
        if col != "Country"
    ]

    data[mbti_columns] = data[mbti_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    data[mbti_columns] = data[mbti_columns].fillna(0)

    return data, mbti_columns


try:
    df, mbti_columns = load_data()

except FileNotFoundError:

    st.error(
        "CSV 파일을 찾을 수 없습니다. "
        "프로젝트 최상위 폴더에 "
        "countriesMBTI_16types.csv 파일이 있는지 확인해주세요."
    )

    st.stop()


# ==========================================
# 페이지 제목
# ==========================================

st.title("🌍 국가별 MBTI 분석")

st.write(
    "국가를 선택하면 해당 국가의 MBTI 16가지 유형별 비율을 "
    "막대그래프로 확인할 수 있어요."
)


# ==========================================
# 국가 선택
# ==========================================

countries = sorted(df["Country"].astype(str).tolist())

default_index = (
    countries.index("South Korea")
    if "South Korea" in countries
    else 0
)

selected_country = st.selectbox(
    "🌎 국가 선택",
    countries,
    index=default_index
)


# ==========================================
# 선택 국가 데이터
# ==========================================

country_data = df[
    df["Country"] == selected_country
].iloc[0]


chart_data = pd.DataFrame({
    "MBTI": mbti_columns,
    "비율": [
        float(country_data[mbti]) * 100
        for mbti in mbti_columns
    ]
})


# 높은 비율부터 정렬
chart_data = chart_data.sort_values(
    "비율",
    ascending=False
).reset_index(drop=True)


# ==========================================
# 색상 그라데이션
# ==========================================

def create_colors(count):

    colors = []

    # 1위 빨간색
    colors.append("#E53935")

    # 나머지 파란색 그라데이션
    dark_blue = (25, 118, 210)
    light_blue = (187, 222, 251)

    for i in range(1, count):

        t = (i - 1) / max(count - 2, 1)

        r = int(
            dark_blue[0]
            + (light_blue[0] - dark_blue[0]) * t
        )

        g = int(
            dark_blue[1]
            + (light_blue[1] - dark_blue[1]) * t
        )

        b = int(
            dark_blue[2]
            + (light_blue[2] - dark_blue[2]) * t
        )

        colors.append(
            f"rgb({r},{g},{b})"
        )

    return colors


colors = create_colors(len(chart_data))


# ==========================================
# 그래프 제목
# ==========================================

st.subheader(
    f"📊 {selected_country}의 MBTI 비율"
)


# ==========================================
# 막대그래프 생성
# ==========================================

max_value = chart_data["비율"].max()

if max_value <= 0:
    max_value = 1


graph_html = """
<div style="
    width:100%;
    font-family:Arial,sans-serif;
    padding:10px 0;
">
"""


for i, row in chart_data.iterrows():

    mbti = html.escape(str(row["MBTI"]))

    value = float(row["비율"])

    color = colors[i]

    width = (value / max_value) * 100

    graph_html += f"""
    <div style="
        display:flex;
        align-items:center;
        gap:12px;
        margin-bottom:14px;
        width:100%;
    ">

        <div style="
            width:55px;
            min-width:55px;
            font-size:14px;
            font-weight:bold;
            color:#333;
        ">
            {mbti}
        </div>

        <div style="
            flex:1;
            height:27px;
            background:#f0f2f5;
            border-radius:5px;
            overflow:hidden;
        ">

            <div style="
                width:{width:.2f}%;
                height:27px;
                background:{color};
                border-radius:5px;
            ">
            </div>

        </div>

        <div style="
            width:70px;
            min-width:70px;
            text-align:right;
            font-size:13px;
            color:#333;
        ">
            {value:.2f}%
        </div>

    </div>
    """


graph_html += "</div>"


# 그래프 표시
st.markdown(
    graph_html,
    unsafe_allow_html=True
)


# ==========================================
# 1위 유형 표시
# ==========================================

top_mbti = chart_data.iloc[0]

st.success(
    f"🏆 {selected_country}의 1위 MBTI는 "
    f"{top_mbti['MBTI']}이며, 비율은 "
    f"{top_mbti['비율']:.2f}%입니다."
)


# ==========================================
# 데이터 표
# ==========================================

with st.expander("📋 MBTI 비율 데이터 보기"):

    table_data = chart_data.copy()

    table_data["비율"] = table_data["비율"].apply(
        lambda x: f"{x:.2f}%"
    )

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 안내 문구
# ==========================================

st.caption(
    "※ 국가별 MBTI 비율은 참고용 통계이며, "
    "개인의 성격을 단정하는 자료가 아닙니다."
)
