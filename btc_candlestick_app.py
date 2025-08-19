
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import time
from dotenv import load_dotenv

# --- 페이지 및 환경 설정 ---
st.set_page_config(layout="wide")
st.title("모델별 예측 결과 시각화 대시보드")

load_dotenv()

# 환경 변수에서 서버 주소 가져오기 (없으면 기본값 사용)
INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL", "http://localhost:8000")
REGISTERED_MODEL_NAME = "BTC_LSTM_Production"

# --- 사이드바 UI ---
st.sidebar.title("📈 모델 예측 제어")

# --- 모델 별칭 한 번만 받아오기 ---
def get_model_aliases():
    """ GET '/aliases' api로 alias 가져오기 """
    try:
        response = requests.get(f"{INFERENCE_SERVER_URL}/aliases", timeout=120)
        response.raise_for_status()
        data = response.json()  # dict
        return data.get("aliases", [])
    except Exception as e:
        st.sidebar.error(f"alias 연결 실패: {e}")
        return ["backtest_20250531"]

# 앱 시작 시 한 번만 실행
model_aliases = get_model_aliases()
model_aliases_prefix = list(map(lambda x: x.removeprefix('backtest_'), model_aliases))

# 챔피언 모델의 예측치 보여주는 버튼
champion_button = st.sidebar.button("Champion Model 예측")

selected_alias = st.sidebar.selectbox(
    "예측 기준 모델(Alias)을 선택하세요:",
    model_aliases_prefix,
    help="이 모델이 학습된 날짜 이후의 기간을 예측합니다."
)

predict_button = st.sidebar.button(
    "선택한 모델로 예측 생성",
    disabled=not selected_alias
)

clear_button = st.sidebar.button("예측 결과 모두 지우기")

st.sidebar.markdown("--- ")
# --- 메인 대시보드 ---

# 추론 서버에서 데이터 가져오기

# api에서 가져오는 로직
# 추론 서버에 alias 요청
api_endpoint = f"{INFERENCE_SERVER_URL}/btc-info"
btc_response = requests.get(api_endpoint, timeout=120)

ohlcv_data = btc_response.json()
ohlcv_df = pd.DataFrame(ohlcv_data)
ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['datetime'])

# 예측 결과를 세션 상태에 저장하기 위한 초기화 (딕셔너리 형태)
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# 예측 결과 지우기 버튼 로직
if clear_button:
    st.session_state.predictions = {}

# 챔피언 예측 생성 버튼 로직
if champion_button:
    with st.spinner("Champion 모델로 예측을 생성합니다..."):
        try:
            api_endpoint = f"{INFERENCE_SERVER_URL}/predict-champion"
            response = requests.get(api_endpoint, timeout=300)
            response.raise_for_status()
            pred_data = response.json()

            # JSON 그대로 DataFrame으로 변환
            prediction_df = pd.DataFrame(pred_data['predictions'])  # already has 'datetime' & 'prediction'

            st.session_state.predictions["champion_model"] = prediction_df
            st.success("✅ Champion 모델 예측 성공!")

        except Exception as e:
            st.error(f"Champion 예측 호출 실패: {e}")

if predict_button:
    with st.spinner(f"'{selected_alias}' 모델 기준으로 예측을 생성합니다..."):
        try:
            # 모델 리로드
            reload_endpoint = f"{INFERENCE_SERVER_URL}/reload?alias={selected_alias}"
            requests.post(reload_endpoint, timeout=120).raise_for_status()
            time.sleep(5)

            # 예측 요청
            predict_endpoint = f"{INFERENCE_SERVER_URL}/predict?alias={selected_alias}"
            response = requests.post(predict_endpoint, timeout=120)
            response.raise_for_status()
            pred_data = response.json()

            # JSON 그대로 DataFrame으로 변환
            prediction_df = pd.DataFrame(pred_data['predictions'])  # already has 'datetime' & 'prediction'

            # 'datetime' 컬럼을 datetime 타입으로 변환
            prediction_df['datetime'] = pd.to_datetime(prediction_df['datetime'])

            # ohlcv_df 마지막 날짜까지만 필터링
            last_data_date = ohlcv_df['datetime'].max()
            prediction_df = prediction_df[prediction_df['datetime'] <= last_data_date].reset_index(drop=True)

            # 세션 상태 저장
            st.session_state.predictions[selected_alias] = prediction_df
            st.success(f"✅ '{selected_alias}' 모델 예측 완료! (총 {len(prediction_df)}시간 예측)")

        except (requests.exceptions.RequestException, KeyError) as e:
            st.warning(f"API 호출 실패 ({e})")

# --- 차트 그리기 ---
fig = go.Figure()

# 1. 실제 가격 캔들스틱 차트
fig.add_trace(go.Candlestick(x=ohlcv_df['datetime'],
                             open=ohlcv_df['btc_open'],
                             high=ohlcv_df['btc_high'],
                             low=ohlcv_df['btc_low'],
                             close=ohlcv_df['btc_close'],
                             name='실제 가격'))

# 2. 저장된 모든 예측값을 순회하며 라인 차트로 추가
if st.session_state.predictions:
    colors = ['orange', 'purple', 'green', 'cyan', 'magenta', 'yellow']
    color_idx = 0
    for alias, pred_df in st.session_state.predictions.items():
        fig.add_trace(go.Scatter(
            x=pred_df['datetime'],
            y=pred_df['prediction'],
            mode='lines',
            line=dict(color=colors[color_idx % len(colors)], width=3),
            name=f'{alias} 모델 예측'
        ))
        color_idx += 1

# --- y축 범위 재계산 (실제 가격 + 모든 예측값 포함) ---
all_prices = ohlcv_df[['btc_low', 'btc_high']].stack()
for pred_df in st.session_state.predictions.values():
    all_prices = pd.concat([all_prices, pred_df['prediction']])

min_price = all_prices.min()
max_price = all_prices.max()
padding = (max_price - min_price) * 0.05

# 차트 레이아웃 업데이트
fig.update_layout(
    title='BTC/USD Candlestick Chart & Model Prediction',
    yaxis_title='Price (USD)',
    xaxis_title='Date',
    xaxis_rangeslider_visible=False,
    yaxis_range=[min_price - padding, max_price + padding],
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# 데이터 테이블
st.subheader("원본 데이터 미리보기")
st.dataframe(ohlcv_df.tail())
