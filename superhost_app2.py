import streamlit as st
import pandas as pd
import joblib

# --- 길이 기준 (실제 값으로 대체 필요) ---
HOST_ABOUT_MEDIAN = 94      # 예시값, 실제 데이터에서 추출 필요
NAME_MEDIAN = 33            # 예시값, 실제 데이터에서 추출 필요
DESCRIPTION_MEAN = 337      # 예시값, 실제 데이터에서 추출 필요

def group_host_about_length(length):
    if length == 0:
        return 'empty'
    elif length > HOST_ABOUT_MEDIAN:
        return 'long'
    else:
        return 'short_or_med'

def group_name_length(length):
    if length == 0:
        return 'empty'
    elif length > NAME_MEDIAN:
        return 'long'
    else:
        return 'short_or_med'

def group_description_length(length):
    if length == 0:
        return 'empty'
    elif length > DESCRIPTION_MEAN:
        return 'long'
    else:
        return 'short_or_avg'

def response_time_to_score(response_time_str):
    mapping = {
        'within an hour': 4,
        'within a few hours': 3,
        'within a day': 2,
        'a few days or more': 1
    }
    return mapping.get(response_time_str.lower(), 0)

def response_rate_to_score(rate_percent):
    rate = rate_percent / 100
    if rate <= 0.25:
        return 1
    elif rate <= 0.5:
        return 2
    elif rate <= 0.75:
        return 3
    else:
        return 4

def acceptance_rate_to_score(rate_percent):
    rate = rate_percent / 100
    if rate <= 0.25:
        return 1
    elif rate <= 0.5:
        return 2
    elif rate <= 0.75:
        return 3
    else:
        return 4

common_amenities = ['Carbon monoxide alarm', 'Essentials', 'Hangers', 'Smoke alarm', 'Wifi']
type_amenity_dict = {
    'high': ['Air conditioning', 'Building staff', 'Elevator', 'Gym', 'Heating', 'Paid parking off premises', 'Shampoo'],
    'low-mid': ['Cleaning products', 'Dining table', 'Exterior security cameras on property', 'Free street parking',
                'Freezer', 'Laundromat nearby', 'Lock on bedroom door', 'Microwave'],
    'mid': ['Cooking basics', 'Kitchen', 'Oven'],
    'upper-mid': ['Bathtub', 'Cleaning products', 'Cooking basics', 'Dishes and silverware', 'Elevator', 'Freezer']
}

def calc_amenity_scores(amenities_list, room_new_type):
    common_match = sum(1 for a in amenities_list if a in common_amenities) / len(common_amenities) if common_amenities else 0
    type_amenities = type_amenity_dict.get(room_new_type, [])
    type_match = sum(1 for a in amenities_list if a in type_amenities) / len(type_amenities) if type_amenities else 0
    return round(common_match, 3), round(type_match, 3)

# --- 모델 불러오기 ---
try:
    pipeline = joblib.load('superhost_pipeline_rf.pkl')
except FileNotFoundError:
    st.error("오류: 'superhost_pipeline_rf.pkl' 모델 파일을 찾을 수 없습니다. 경로를 확인하거나 파일을 스크립트와 같은 디렉토리에 두세요.")
    st.stop()

st.set_page_config(layout="wide")
st.title("🌟 Airbnb Superhost 🌟")
st.markdown("---")

st.write("**입력값은 실제 모델 학습 데이터와 1:1로 일치합니다.**")
st.info("호스트 소개글, 숙소 이름, 숙소 설명은 글자수로 입력하면 자동으로 그룹화됩니다.")

def main():
    st.header("🏠 숙소 및 호스트 정보 입력")

    # 글자수 입력 (설명은 help 파라미터로 제공)
    host_about_length = st.number_input(
        f"호스트 소개글 길이 (글자수)",
        min_value=0, value=100,
        help=f"0이면 없음, {HOST_ABOUT_MEDIAN}자 초과: long (모델 기준 long이 유리, 실제로는 상세한 소개글이 신뢰를 높입니다)"
    )
    name_length = st.number_input(
        f"숙소 이름 길이 (글자수)",
        min_value=0, value=35,
        help=f"0이면 없음, {NAME_MEDIAN}자 초과: long (모델 기준 long이 유리, 실제로는 명확하고 긴 이름이 더 매력적입니다)"
    )
    description_length = st.number_input(
        f"숙소 설명 길이 (글자수)",
        min_value=0, value=350,
        help=f"0이면 없음, {DESCRIPTION_MEAN}자 초과: long (모델 기준 short_or_avg가 유리, 실제로는 상세한 설명이 더 신뢰를 줍니다)"
    )

    host_about_length_group = group_host_about_length(host_about_length)
    name_length_group = group_name_length(name_length)
    description_length_group = group_description_length(description_length)

    room_type = st.selectbox("숙소 유형", ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'], index=0)
    host_has_profile_pic = st.radio("프로필 사진 유무", [1, 0], format_func=lambda x: "있음" if x == 1 else "없음", index=0)
    host_identity_verified = st.radio("호스트 신원 인증", [1, 0], format_func=lambda x: "예" if x == 1 else "아니오", index=0)
    is_long_term = st.radio("장기 숙박 가능 여부", [1, 0], format_func=lambda x: "예" if x == 1 else "아니오", index=0)
    accommodates = st.number_input("최대 수용 인원", min_value=1, value=2)

    # 편의시설
    st.subheader("편의시설 선택")
    all_possible_amenities = sorted(list(set(common_amenities +
                                          type_amenity_dict['high'] +
                                          type_amenity_dict['low-mid'] +
                                          type_amenity_dict['mid'] +
                                          type_amenity_dict['upper-mid'] +
                                          ['TV', 'Dryer', 'Washer', 'Dishwasher', 'Coffee maker', 'Toaster', 'Iron', 'Hair dryer',
                                           'Bed linens', 'Extra pillows and blankets', 'First aid kit', 'Fire extinguisher', 'Locker',
                                           'Pillow', 'Laptop friendly workspace', 'Hot water', 'Heating', 'Air conditioning', 'Shampoo', 'Cooking basics', 'Kitchen', 'Oven',
                                           'Essentials', 'Hangers', 'Smoke alarm', 'Wifi', 'Carbon monoxide alarm'
                                          ])))
    selected_amenities = st.multiselect(
        "제공하는 편의시설을 선택하세요 (다중 선택 가능)",
        options=all_possible_amenities,
        default=common_amenities,
        help=f"**슈퍼호스트는 평균 22개 편의시설을 제공합니다.**"
    )
    amenities_cnt = len(selected_amenities)
    st.write(f"선택된 편의시설 개수: **{amenities_cnt}**")

    # 가격대 그룹(편의시설 점수 계산용)
    room_new_type = st.selectbox("숙소 가격대 그룹 (편의시설 점수 계산용)", ['high', 'low-mid', 'mid', 'upper-mid'], index=2)
    common_amenity_score, type_amenity_score = calc_amenity_scores(selected_amenities, room_new_type)

    # 기타 수치형 입력
    availability_365 = st.number_input("연간 예약 가능일 수 (최대 365일)", min_value=0, max_value=365, value=200)
    price = st.number_input("1박당 가격 ($)", min_value=0, value=53)
    host_response_time = st.selectbox("호스트 응답 시간", ['within an hour', 'within a few hours', 'within a day', 'a few days or more'], index=0)
    host_response_rate = st.slider("호스트 응답률 (%)", 0, 100, 100)
    host_acceptance_rate = st.slider("호스트 수락률 (%)", 0, 100, 100)

    # 점수 계산
    response_time_score = response_time_to_score(host_response_time)
    response_rate_score = response_rate_to_score(host_response_rate)
    acceptance_rate_score = acceptance_rate_to_score(host_acceptance_rate)

    # 입력 데이터프레임 생성
    input_data_dict = {
        'amenities_cnt': amenities_cnt,
        'availability_365': availability_365,
        'price': price,
        'host_about_length_group': host_about_length_group,
        'room_type': room_type,
        'name_length_group': name_length_group,
        'description_length_group': description_length_group,
        'host_has_profile_pic': host_has_profile_pic,
        'host_response_time_score': response_time_score,
        'type_amenity_score': type_amenity_score,
        'common_amenity_score': common_amenity_score,
        'host_acceptance_rate_score': acceptance_rate_score,
        'host_identity_verified': host_identity_verified,
        'is_long_term': is_long_term,
        'accommodates': accommodates
    }
    new_data_df = pd.DataFrame([input_data_dict])

    st.markdown("---")
    if st.button("✨ 슈퍼호스트 확률 예측하기 ✨"):
        try:
            pred = pipeline.predict(new_data_df)
            proba = pipeline.predict_proba(new_data_df)[:, 1]

            st.subheader("💡 예측 결과")
            if pred[0] == 1:
                st.success(f"🎉 **슈퍼호스트가 될 가능성이 높습니다!**")
                st.markdown(f"**예측 확률: <span style='color:green; font-size:2em;'>{round(proba[0]*100, 2)}%</span>**", unsafe_allow_html=True)
            else:
                st.warning(f"🤔 **현재 조건으로는 슈퍼호스트가 아닐 가능성이 높습니다.**")
                st.markdown(f"**예측 확률: <span style='color:orange; font-size:2em;'>{round(proba[0]*100, 2)}%</span>**", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### [모델 인사이트]")
            st.info(
                "모델은 실제 데이터에서 '소개글이 empty', '설명 길이가 short_or_avg', '장기 숙박 가능', '가격이 낮음', '수용 인원 1명' 등 "
                "사람의 직관과 다를 수 있는 조합에서 가장 높은 확률을 내기도 합니다. "
                "이는 데이터에 기반한 결과이므로, 실제 슈퍼호스트가 되기 위해서는 상세한 소개글, 다양한 편의시설, 높은 응답률 등도 중요합니다."
            )
            st.markdown("""
            <ul>
                <li>✔️ <b>모델이 중요하게 생각하는 feature 조합</b>을 실험해보세요.</li>
                <li>✔️ <b>입력값을 다양하게 바꿔보며</b> 예측 확률이 어떻게 변하는지 확인해보세요.</li>
                <li>✔️ <b>모델의 예측 결과는 참고용</b>이며, 실제 슈퍼호스트 선정에는 다양한 요소가 반영됩니다.</li>
            </ul>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {e}")
            st.write("입력 데이터의 컬럼과 모델 학습 시의 컬럼이 일치하는지 확인이 필요합니다.")

if __name__ == "__main__":
    main()
