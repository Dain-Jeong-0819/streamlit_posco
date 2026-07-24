# AI 국내 주식 분석 챗봇

`yfinance`로 국내 주식 일봉 데이터를 조회하고, `gpt-4o-mini`가 해당
데이터를 바탕으로 답변하는 Streamlit 챗봇입니다.

## 주요 기능

- 국내 KOSPI·KOSDAQ 종목코드 조회
- 기간별 종가 차트와 OHLCV 데이터 표시
- 최근 종가, 기간 수익률, 최고가, 평균 거래량 계산
- 조회한 종목 데이터를 근거로 하는 AI 스트리밍 답변
- 종목 변경 시 이전 대화 자동 초기화

> Yahoo Finance 데이터는 지연되거나 부정확할 수 있으며 개인적인 연구 및
> 교육 목적으로만 사용해야 합니다. 이 앱의 답변은 투자 자문이 아닙니다.

## 로컬 실행

1. 패키지를 설치합니다.

   ```bash
   pip install -r requirements.txt
   ```

2. `.streamlit/secrets.toml` 파일을 만들고 API 키를 입력합니다.

   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

3. 앱을 실행합니다.

   ```bash
   streamlit run app.py
   ```

## Streamlit Community Cloud 배포

1. 이 폴더의 파일을 GitHub 저장소에 올립니다.
2. [Streamlit Community Cloud](https://share.streamlit.io/)에서 저장소와
   `app.py`를 선택합니다.
3. **Advanced settings → Secrets**에 아래 내용을 등록합니다.

   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

4. **Deploy**를 누릅니다.

API 키가 들어 있는 `secrets.toml` 파일은 Git에 커밋하지 마세요.
