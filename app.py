import os

import streamlit as st
from openai import OpenAI


MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "당신은 친절하고 정확한 AI 어시스턴트입니다. "
    "사용자의 언어에 맞춰 명확하고 도움이 되도록 답변하세요."
)


st.set_page_config(
    page_title="OpenAI 챗봇",
    page_icon="💬",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 820px; padding-top: 2.2rem;}
        [data-testid="stChatMessage"] {
            border: 1px solid rgba(128, 128, 128, 0.16);
            border-radius: 16px;
            padding: 0.35rem 0.8rem;
        }
        .app-subtitle {color: #6b7280; margin-top: -0.65rem;}
        .model-badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: rgba(16, 163, 127, 0.12);
            color: #0b8066;
            font-size: 0.82rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key() -> str | None:
    """Read the key from Streamlit secrets, with an env fallback for local use."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (FileNotFoundError, KeyError):
        return os.getenv("OPENAI_API_KEY")


def stream_answer(client: OpenAI, messages: list[dict[str, str]]):
    """Yield text fragments from the OpenAI streaming response."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text


st.title("💬 OpenAI 챗봇")
st.markdown(
    '<p class="app-subtitle">궁금한 내용을 입력하면 AI가 실시간으로 답변합니다.</p>',
    unsafe_allow_html=True,
)
st.markdown(f'<span class="model-badge">{MODEL}</span>', unsafe_allow_html=True)

with st.sidebar:
    st.header("챗봇 설정")
    st.caption("사용 모델")
    st.code(MODEL, language=None)
    st.divider()
    if st.button("🗑️ 대화 내용 지우기", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("API 키는 Streamlit Secrets에서 안전하게 불러옵니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

api_key = get_api_key()
if not api_key:
    st.warning(
        "OpenAI API 키가 설정되지 않았습니다. "
        "Streamlit Community Cloud의 **App settings → Secrets**에 "
        '`OPENAI_API_KEY = "sk-..."` 형식으로 등록해 주세요.'
    )
    st.stop()

client = OpenAI(api_key=api_key)

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("안녕하세요! 무엇을 도와드릴까요? 👋")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                stream_answer(client, st.session_state.messages)
            )
        except Exception as error:
            st.error(
                "답변을 생성하지 못했습니다. API 키, 사용 한도, 네트워크 상태를 "
                f"확인해 주세요.\n\n오류: `{error}`"
            )
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
