import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="월별 기온분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 월별 기온분석")

# 한글 인코딩 처리
df = pd.read_csv("temperature_cleaned.csv", encoding="utf-8-sig")

# 데이터 형식 변환
df["연도"] = pd.to_numeric(df["연도"], errors="coerce")
df["월번호"] = pd.to_numeric(df["월번호"], errors="coerce")
df["최고기온(℃)"] = pd.to_numeric(
    df["최고기온(℃)"], errors="coerce"
)
df["최저기온(℃)"] = pd.to_numeric(
    df["최저기온(℃)"], errors="coerce"
)

df = df.dropna(
    subset=["연도", "월번호", "최고기온(℃)", "최저기온(℃)"]
)

df["연도"] = df["연도"].astype(int)
df["월번호"] = df["월번호"].astype(int)

# 월 선택
month_names = {
    1: "1월", 2: "2월", 3: "3월", 4: "4월",
    5: "5월", 6: "6월", 7: "7월", 8: "8월",
    9: "9월", 10: "10월", 11: "11월", 12: "12월"
}

selected_month = st.selectbox(
    "분석할 월을 선택하세요.",
    list(month_names.keys()),
    format_func=lambda x: month_names[x]
)

# 선택한 월의 데이터 추출
monthly = df[df["월번호"] == selected_month].copy()

# 연도별 최고기온과 최저기온 집계
monthly = (
    monthly.groupby("연도")
    .agg({
        "최고기온(℃)": "max",
        "최저기온(℃)": "min"
    })
    .reset_index()
    .sort_values("연도")
)

# 그래프 생성
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    monthly["연도"],
    monthly["최고기온(℃)"],
    color="hotpink",
    marker="o",
    linewidth=2,
    label="최고기온"
)

ax.plot(
    monthly["연도"],
    monthly["최저기온(℃)"],
    color="skyblue",
    marker="o",
    linewidth=2,
    label="최저기온"
)

# 그래프 제목 및 축
ax.set_title("월별 기온분석", fontsize=20)
ax.set_xlabel("연도", fontsize=13)
ax.set_ylabel("온도 (℃)", fontsize=13)

# 범례 및 격자
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

# 연도 표시
step = max(1, len(monthly) // 15)
ax.set_xticks(monthly["연도"].iloc[::step])
plt.xticks(rotation=45)

plt.tight_layout()

# Streamlit 출력
st.pyplot(fig, use_container_width=True)

# 데이터 표
st.subheader(f"📊 {month_names[selected_month]} 연도별 기온 데이터")
st.dataframe(monthly, use_container_width=True, hide_index=True)
