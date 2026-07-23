# OpenAI Streamlit 챗봇

`gpt-4o-mini`와 OpenAI API를 사용하는 스트리밍 챗봇입니다.

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
