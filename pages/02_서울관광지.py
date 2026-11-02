
import streamlit as st
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="서울 관광지 TOP 10",
    page_icon="🗺️",
    layout="wide"
)

# 관광지 데이터
tourist_spots = [
    {
        "name": "경복궁",
        "english": "Gyeongbokgung Palace",
        "lat": 37.5796,
        "lon": 126.9770,
        "description": "조선 왕조의 대표 궁궐이에요. 한복을 입고 방문하면 특별한 추억을 만들 수 있어요.",
        "category": "🏯 역사"
    },
    {
        "name": "명동",
        "english": "Myeongdong",
        "lat": 37.5636,
        "lon": 126.9869,
        "description": "쇼핑과 길거리 음식으로 유명한 곳이에요. 다양한 한국 화장품과 간식을 만나보세요.",
        "category": "🛍️ 쇼핑"
    },
    {
        "name": "남산서울타워",
        "english": "N Seoul Tower",
        "lat": 37.5512,
        "lon": 126.9882,
        "description": "서울의 아름다운 전경을 감상할 수 있는 전망 명소예요. 야경도 멋지답니다.",
        "category": "🌃 전망"
    },
    {
        "name": "북촌한옥마을",
        "english": "Bukchon Hanok Village",
        "lat": 37.5826,
        "lon": 126.9830,
        "description": "전통 한옥이 모여 있는 아름다운 마을이에요. 한국의 옛 분위기를 느껴보세요.",
        "category": "🏡 전통"
    },
    {
        "name": "인사동",
        "english": "Insadong",
        "lat": 37.5740,
        "lon": 126.9855,
        "description": "전통 공예품과 기념품을 구경하기 좋은 거리예요. 전통차와 한국 문화를 즐겨보세요.",
        "category": "🎨 문화"
    },
    {
        "name": "홍대",
        "english": "Hongdae",
        "lat": 37.5563,
        "lon": 126.9236,
        "description": "젊고 활기찬 분위기의 거리예요. 음악, 공연, 카페, 쇼핑을 즐길 수 있어요.",
        "category": "🎸 문화"
    },
    {
        "name": "동대문디자인플라자",
        "english": "DDP",
        "lat": 37.5669,
        "lon": 127.0096,
        "description": "독특한 건축 디자인으로 유명한 공간이에요. 전시와 패션 행사도 열려요.",
        "category": "🏙️ 건축"
    },
    {
        "name": "광장시장",
        "english": "Gwangjang Market",
        "lat": 37.5700,
        "lon": 126.9997,
        "description": "빈대떡, 김밥 등 다양한 한국 음식을 맛볼 수 있는 전통 시장이에요.",
        "category": "🍜 먹거리"
    },
    {
        "name": "롯데월드타워",
        "english": "Lotte World Tower",
        "lat": 37.5125,
        "lon": 127.1025,
        "description": "서울의 대표적인 초고층 건물이에요. 전망대에서 멋진 도시 풍경을 감상해 보세요.",
        "category": "🏙️ 전망"
    },
    {
        "name": "청계천",
        "english": "Cheonggyecheon Stream",
        "lat": 37.5692,
        "lon": 126.9788,
        "description": "서울 도심을 따라 흐르는 산책 명소예요. 도심 속에서 여유롭게 걸어보세요.",
        "category": "🌿 자연"
    }
]

# 제목
st.title("🗺️ 서울 관광지 TOP 10")
st.subheader("외국인 관광객에게 인기 있는 서울의 대표 관광 명소")
st.write("서울 여행을 계획하고 있다면 아래 지도에서 가보고 싶은 장소를 찾아보세요! ✈️")

st.divider()

# 지도용 데이터
map_data = pd.DataFrame([
    {
        "lat": spot["lat"],
        "lon": spot["lon"]
    }
    for spot in tourist_spots
])

# 지도 표시
st.subheader("📍 서울 주요 관광지 지도")

st.map(
    map_data,
    latitude="lat",
    longitude="lon",
    zoom=11,
    height=500
)

st.caption("※ 지도에는 관광지 10곳의 위치가 표시됩니다. 마커는 실제 관광지 위치를 기준으로 합니다.")

st.divider()

# 관광지 설명
st.subheader("🎒 관광지 소개")

# 2열 카드 형식
col1, col2 = st.columns(2)

for i, spot in enumerate(tourist_spots):
    with col1 if i % 2 == 0 else col2:
        with st.container(border=True):
            st.markdown(f"### {i + 1}. {spot['name']}")
            st.caption(spot["english"])
            st.markdown(spot["category"])
            st.write(spot["description"])

st.divider()

st.info(
    "💡 여행 팁: 경복궁, 북촌한옥마을, 인사동은 서로 가까워 "
    "하루 일정으로 함께 둘러보기 좋아요!"
)

st.caption("서울 관광지 소개 앱 | 서울 여행을 즐겨보세요 🇰🇷")
