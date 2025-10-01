
import streamlit as st
import pandas as pd
import joblib

# --- 점수 변환 함수들 ---
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
    # 모델 파일이 현재 스크립트와 같은 디렉토리에 있다고 가정합니다.
    # 만약 다른 경로라면 'C:/Users/HY/Documents/GitHub/advanced_project/hayoung/3/superhost_pipeline_rf.pkl' 처럼 절대 경로를 다시 지정해야 합니다.
    pipeline = joblib.load('superhost_pipeline_rf.pkl')
    # pipeline에서 실제 train_columns를 추출할 수 있다면 더 좋습니다 (예: pipeline.named_steps['preprocessor'].get_feature_names_out())
    # 하지만 여기서는 입력 DataFrame이 pipeline에 바로 들어가는 형태이므로, 컬럼명은 데이터프레임 생성 시 맞춰주면 됩니다.
except FileNotFoundError:
    st.error("오류: 'superhost_pipeline_rf.pkl' 모델 파일을 찾을 수 없습니다. 경로를 확인하거나 파일을 스크립트와 같은 디렉토리에 두세요.")
    st.stop() # 파일이 없으면 앱 실행 중지

st.set_page_config(layout="wide") # 페이지 레이아웃을 넓게 설정
st.title("🌟 Airbnb Superhost 🌟")
st.markdown("---") # 구분선 추가

st.write("**당신의 숙소와 호스트 정보를 입력하여 슈퍼호스트가 될 가능성을 예측해보세요!**")
st.info("💡 슈퍼호스트는 에어비앤비의 특정 기준(응답률, 수락률, 평점, 예약 건수 등)을 충족해야 부여되는 자격입니다.")

def main():
    st.header("🏠 숙소 정보")
    
    # 숙소 정보 입력 위젯들
    room_new_type = st.selectbox("숙소 가격대 그룹", ['high', 'low-mid', 'mid', 'upper-mid'], index=2, help="숙소의 대략적인 가격대. 'mid'가 슈퍼호스트에게 선호되는 경향이 있습니다.")
    room_type = st.selectbox("숙소 유형", ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'], index=0, help="'Entire home/apt'(집 전체/아파트)가 슈퍼호스트에게 유리합니다.")
    
    # 모든 가능한 편의시설을 리스트로 합치기 (다양한 편의시설을 추가하여 기본값을 높임)
    all_possible_amenities = sorted(list(set(common_amenities +
                                          type_amenity_dict['high'] +
                                          type_amenity_dict['low-mid'] +
                                          type_amenity_dict['mid'] +
                                          type_amenity_dict['upper-mid'] +
                                          ['TV', 'Dryer', 'Washer', 'Dishwasher', 'Coffee maker', 'Toaster', 'Iron', 'Hair dryer',
                                           'Bed linens', 'Extra pillows and blankets', 'First aid kit', 'Fire extinguisher', 'Locker',
                                           'Pillow', 'Laptop friendly workspace', 'Hot water', 'Heating', 'Air conditioning', 'Shampoo', 'Cooking basics', 'Kitchen', 'Oven',
                                           'Essentials', 'Hangers', 'Smoke alarm', 'Wifi', 'Carbon monoxide alarm' # 중복 방지
                                          ])))
    
    # 기본 선택될 편의시설 설정 (슈퍼호스트에 유리한 조건으로 최대한 많이 선택)
    default_amenities = [
        'Carbon monoxide alarm', 'Essentials', 'Hangers', 'Smoke alarm', 'Wifi', # 공통 필수
        'Cooking basics', 'Kitchen', 'Oven', # mid 타입 필수
        'Air conditioning', 'Heating', 'Shampoo', # high/mid/upper-mid
        'TV', 'Dryer', 'Washer', 'Dishwasher', 'Coffee maker', 'Toaster', 'Iron', 'Hair dryer',
        'Bed linens', 'Extra pillows and blankets', 'First aid kit', 'Fire extinguisher', 'Locker',
        'Pillow', 'Laptop friendly workspace', 'Hot water' # 기타 중요
    ]
    # 실제 존재하는 옵션들만 기본값으로 설정
    default_amenities = [a for a in default_amenities if a in all_possible_amenities]


    selected_amenities = st.multiselect(
        "제공하는 편의시설을 선택하세요 (다중 선택 가능)",
        options=all_possible_amenities,
        default=default_amenities, # 기본으로 선택될 값 조정
        help=f"**슈퍼호스트는 평균 37개 이상의 편의시설을 제공합니다.** 현재 선택된 개수: {len(default_amenities)}개."
    )
    amenities_cnt = len(selected_amenities) # 선택된 편의시설 개수를 자동으로 반영
    st.write(f"➡️ 선택된 편의시설 개수: **{amenities_cnt}**")

    availability_365 = st.number_input("연간 예약 가능일 수 (최대 365일)", min_value=0, max_value=365, value=300, help="1년 중 숙소를 예약 가능한 일수. 슈퍼호스트는 평균 233일 이상으로 높은 가용성을 보입니다.") # 300으로 변경
    price = st.number_input("1박당 가격 ($)", min_value=0, value=110, help="숙소의 1박당 가격. 슈퍼호스트는 평균 $129보다 약간 낮은 가격대를 유지하는 경향이 있습니다.") # 110으로 변경
    accommodates = st.number_input("최대 수용 인원", min_value=1, value=2, help="숙소에서 수용 가능한 최대 게스트 수. 2명 수용 숙소가 중앙값입니다.") # 2로 변경

    st.header("👤 호스트 및 숙소 정보 상세")
    
    host_response_time = st.selectbox("호스트 응답 시간", ['within an hour', 'within a few hours', 'within a day', 'a few days or more'], index=0, help="게스트 문의에 대한 응답 시간. **1시간 이내**가 슈퍼호스트의 핵심 요건입니다.")
    host_response_rate = st.slider("호스트 응답률 (%)", 0, 100, 100, help="게스트 문의에 응답한 비율. **100%**를 유지하는 것이 매우 중요합니다.") # 100으로 변경
    host_acceptance_rate = st.slider("호스트 수락률 (%)", 0, 100, 100, help="예약 요청을 수락한 비율. **100%**에 가까울수록 좋습니다.") # 100으로 변경
    
    host_about_length_group = st.selectbox("호스트 소개글 길이", ['short', 'medium', 'long'], index=2, help="프로필 소개글의 길이. **길고 상세한 소개글**이 신뢰도를 높여줍니다.") # long (index=2)
    name_length_group = st.selectbox("숙소 이름 길이", ['short', 'medium', 'long'], index=2, help="숙소 이름의 길이. **길고 명확한 이름**이 숙소의 매력을 더 잘 전달합니다.") # long (index=2)
    description_length_group = st.selectbox("숙소 설명 길이", ['short', 'medium', 'long'], index=2, help="숙소 상세 설명의 길이. **길고 자세한 설명**이 게스트의 이해를 돕고 만족도를 높입니다.") # long (index=2)
    
    host_has_profile_pic = st.radio("프로필 사진 유무", [1, 0], format_func=lambda x: "있음" if x == 1 else "없음", index=0, help="프로필 사진 유무. **거의 모든 슈퍼호스트는 프로필 사진을 가지고 있습니다.**")
    host_identity_verified = st.radio("호스트 신원 인증", [1, 0], format_func=lambda x: "예" if x == 1 else "아니오", index=0, help="에어비앤비에서 신원 인증을 했는지 여부. **인증된 호스트**가 더 신뢰를 얻습니다.")
    is_long_term = st.radio("장기 숙박 가능 여부", [0, 1], format_func=lambda x: "아니오" if x == 0 else "예", index=0, help="장기 임대보다 **단기 숙박 중심**으로 운영하는 것이 슈퍼호스트 자격에 유리합니다.")


    st.markdown("---")
    if st.button("✨ 슈퍼호스트 확률 예측하기 ✨"):
        # --- 점수 계산 ---
        response_time_score = response_time_to_score(host_response_time)
        response_rate_score = response_rate_to_score(host_response_rate)
        acceptance_rate_score = acceptance_rate_to_score(host_acceptance_rate)
        common_amenity_score, type_amenity_score = calc_amenity_scores(selected_amenities, room_new_type)

        # --- 예측을 위한 DataFrame 생성 ---
        input_data_dict = {
            'amenities_cnt': amenities_cnt,
            'availability_365': availability_365,
            'price': price,
            'host_about_length_group': host_about_length_group, # 범주형
            'room_type': room_type,                               # 범주형
            'name_length_group': name_length_group,               # 범주형
            'description_length_group': description_length_group, # 범주형
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


        # --- 예측 실행 ---
        try:
            pred = pipeline.predict(new_data_df)
            proba = pipeline.predict_proba(new_data_df)[:, 1]

            st.subheader("💡 예측 결과")
            if pred[0] == 1:
                st.success(f"🎉 **당신은 슈퍼호스트가 될 가능성이 매우 높습니다!**")
                st.markdown(f"**예측 확률: <span style='color:green; font-size:2em;'>{round(proba[0]*100, 2)}%</span>**", unsafe_allow_html=True)
            else:
                st.warning(f"🤔 **아쉽지만 현재 조건으로는 슈퍼호스트가 아닐 가능성이 높습니다.**")
                st.markdown(f"**예측 확률: <span style='color:orange; font-size:2em;'>{round(proba[0]*100, 2)}%</span>**", unsafe_allow_html=True)
            
            st.markdown("""
            <br>
            <p><strong>슈퍼호스트가 되기 위한 핵심 요건:</strong></p>
            <ul>
                <li>✔️ **응답률 90% 이상 & 응답 시간 1시간 이내:** 게스트 문의에 빠르게 응답하세요.</li>
                <li>✔️ **수락률 90% 이상:** 예약 요청을 적극적으로 수락하세요.</li>
                <li>✔️ **높은 평점:** 종합 평점 4.8점 이상 유지를 목표로 하세요.</li>
                <li>✔️ **많은 예약 건수:** 10건 이상의 숙박 완료 및 게스트 수용이 필요합니다.</li>
                <li>✔️ **다양한 편의시설:** 게스트 편의를 위한 필수 편의시설과 추가 시설을 완비하세요.</li>
                <li>✔️ **상세하고 매력적인 숙소 및 호스트 정보:** 사진과 설명을 충분히 제공하여 신뢰를 얻으세요.</li>
            </ul>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다. 입력 데이터 형식을 확인해주세요: {e}")
            st.write("입력된 데이터의 컬럼과 모델 학습 시의 컬럼이 일치하는지 확인이 필요합니다.")
            # 디버깅을 위해 입력 데이터프레임 구조를 출력
            # st.write("입력 데이터 DataFrame:")
            # st.write(new_data_df)


if __name__ == "__main__":
    main()
