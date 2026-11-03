
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path
import html

# 페이지 설정
st.set_page_config(
    page_title="국가별 MBTI 분석",
    page_icon="🌍",
    layout="wide"
)

# ==================================================
# CSV 파일 경로
# pages 폴더의 상위 폴더에 CSV 파일이 있음
# ==================================================

CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "countriesMBTI_16types.csv"
)


# ==================================================
# 데이터 불러오기
# ==================================================

@st.cache_data
def load_data():

    data = pd.read_csv(CSV_PATH)

    if "Country" not in data.columns:
        raise ValueError("CSV 파일에 Country 열이 없습니다.")

    # Country를 제외한 MBTI 유형 열
    mbti_columns = [
        column for column in data.columns
        if column != "Country"
    ]

    # MBTI 비율 숫자로 변환
    data[mbti_columns] = data[mbti_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # 결측값 처리
    data[mbti_columns] = data[mbti_columns].fillna(0)

    return data, mbti_columns


# ==================================================
# 데이터 읽기 오류 처리
# ==================================================

try:

    df, mbti_columns = load_data()

except FileNotFoundError:

    st.error(
        "CSV 파일을 찾을 수 없습니다.\n\n"
        "countriesMBTI_16types.csv 파일이 "
        "프로젝트 최상위 폴더에 있는지 확인해주세요."
    )

    st.stop()

except Exception as error:

    st.error(f"데이터 오류: {error}")

    st.stop()


# ==================================================
# 제목
# ==================================================

st.title("🌍 국가별 MBTI 분석")

st.write(
    "국가를 선택하면 해당 국가의 MBTI 16가지 유형별 비율을 "
    "막대그래프로 확인할 수 있어요."
)


# ==================================================
# 국가 선택
# ==================================================

countries = sorted(
    df["Country"].astype(str).unique().tolist()
)

# South Korea를 기본 국가로 설정
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


# ==================================================
# 선택 국가 데이터
# ==================================================

country_data = df[
    df["Country"].astype(str) == selected_country
].iloc[0]


# MBTI 유형별 비율 계산
chart_data = pd.DataFrame({
    "MBTI": mbti_columns,
    "비율": [
        float(country_data[mbti]) * 100
        for mbti in mbti_columns
    ]
})


# 비율 높은 순서로 정렬
chart_data = chart_data.sort_values(
    by="비율",
    ascending=False
).reset_index(drop=True)


# ==================================================
# 그래프 색상 생성
# ==================================================

def make_colors(count):

    colors = []

    # 1위 빨간색
    colors.append("#E53935")

    # 나머지는 파란색에서 연한 파란색으로
    dark_blue = (21, 101, 192)
    light_blue = (187, 222, 251)

    for i in range(1, count):

        t = (i - 1) / max(count - 2, 1)

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
            f"rgb({r},{g},{b})"
        )

    return colors


colors = make_colors(len(chart_data))


# ==================================================
# 그래프 제목
# ==================================================

st.subheader(
    f"📊 {selected_country}의 MBTI 유형별 비율"
)


# ==================================================
# HTML 막대그래프 생성
# ==================================================

max_value = chart_data["비율"].max()

if max_value <= 0:
    max_value = 1


graph_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

<style>

body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 10px;
    background: transparent;
}

.chart {
    width: 100%;
}

.bar-row {
    display: flex;
    align-items: center;
    width: 100%;
    margin-bottom: 13px;
    gap: 10px;
}

.mbti-label {
    width: 55px;
    min-width: 55px;
    font-weight: bold;
    font-size: 14px;
    color: #333333;
}

.bar-background {
    flex: 1;
    height: 28px;
    background: #f1f3f5;
    border-radius: 5px;
    overflow: hidden;
}

.bar {
    height: 28px;
    border-radius: 5px;
    transition: width 0.4s ease;
}

.value-label {
    width: 65px;
    min-width: 65px;
    text-align: right;
    font-size: 13px;
    color: #333333;
}

</style>
</head>

<body>

<div class="chart">
"""


# ==================================================
# 각 MBTI 막대 생성
# ==================================================

for index, row in chart_data.iterrows():

    mbti = html.escape(str(row["MBTI"]))

    value = float(row["비율"])

    color = colors[index]

    # 막대 길이
    width = (value / max_value) * 100

    graph_html += f"""
    <div class="bar-row">

        <div class="mbti-label">
            {mbti}
        </div>

        <div class="bar-background">

            <div
                class="bar"
                style="
                    width: {width:.2f}%;
                    background: {color};
                "
            ></div>

        </div>

        <div class="value-label">
            {value:.2f}%
        </div>

    </div>
    """


# ==================================================
# HTML 종료
# ==================================================

graph_html += """
</div>

</body>
</html>
"""


# ==================================================
# 그래프 출력
# HTML 컴포넌트에서 실제 그래프로 렌더링
# ==================================================

components.html(
    graph_html,
    height=680,
    scrolling=False
)


# ==================================================
# 가장 높은 비율의 MBTI 유형
# ==================================================

top_type = chart_data.iloc[0]

st.success(
    f"🏆 {selected_country}의 가장 높은 비율은 "
    f"{top_type['MBTI']}이며, "
    f"비율은 {top_type['비율']:.2f}%입니다."
)


# ==================================================
# 데이터 표
# ==================================================

with st.expander("📋 MBTI 비율 데이터 보기"):

    display_data = chart_data.copy()

    display_data["비율"] = display_data["비율"].apply(
        lambda value: f"{value:.2f}%"
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 안내 문구
# ==================================================

st.caption(
    "※ 국가별 MBTI 비율은 참고용 통계이며, "
    "개인의 성격을 단정하는 자료가 아닙니다."
)
