import os
from typing import Any, Iterator

import pandas as pd
import streamlit as st
import yfinance as yf
from openai import OpenAI


MODEL = "gpt-4o-mini"
PERIOD_LABELS = {
    "1개월": "1mo",
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
}
MARKET_SUFFIXES = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}
BASE_SYSTEM_PROMPT = """
당신은 한국 주식 데이터 분석을 돕는 친절하고 신중한 AI 어시스턴트입니다.
아래에 제공된 yfinance 데이터만을 현재 주가 데이터의 근거로 사용하세요.
데이터에 없는 사실이나 실시간 뉴스는 추측하지 말고, 확인할 수 없다고 명시하세요.
수치에는 가능한 한 기준일을 함께 표시하고 핵심 계산 과정을 간단히 설명하세요.
답변은 정보 제공 및 교육 목적이며, 확정적인 매수·매도 지시나 수익 보장을 하지 마세요.
사용자가 다른 종목을 물으면 먼저 사이드바에서 해당 종목을 조회하도록 안내하세요.
""".strip()


st.set_page_config(
    page_title="AI 국내 주식 분석 챗봇",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1180px; padding-top: 2rem;}
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
        .notice {
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background: rgba(245, 158, 11, 0.10);
            color: #92400e;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_api_key() -> str | None:
    """Read the key from Streamlit secrets, with an environment fallback."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (FileNotFoundError, KeyError):
        return os.getenv("OPENAI_API_KEY")


def make_symbol(code: str, market: str) -> str:
    """Convert a six-digit Korean stock code to a Yahoo Finance symbol."""
    cleaned = code.strip().upper()
    if cleaned.endswith((".KS", ".KQ")):
        return cleaned
    if not (cleaned.isdigit() and len(cleaned) == 6):
        raise ValueError("종목코드는 6자리 숫자로 입력해 주세요. 예: 005930")
    return f"{cleaned}{MARKET_SUFFIXES[market]}"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_data(symbol: str, period: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch daily market data and selected company metadata from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    history = ticker.history(
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
        timeout=10,
    )
    if history.empty:
        raise ValueError("조회된 주가 데이터가 없습니다. 종목코드와 시장을 확인해 주세요.")

    history = history[["Open", "High", "Low", "Close", "Volume"]].dropna(
        subset=["Close"]
    )
    history.index = pd.to_datetime(history.index).tz_localize(None)

    try:
        raw_info = ticker.get_info()
    except Exception:
        raw_info = {}

    info = {
        "name": raw_info.get("longName") or raw_info.get("shortName") or symbol,
        "currency": raw_info.get("currency", "KRW"),
        "exchange": raw_info.get("exchange", ""),
        "sector": raw_info.get("sector", ""),
        "industry": raw_info.get("industry", ""),
        "market_cap": raw_info.get("marketCap"),
        "trailing_pe": raw_info.get("trailingPE"),
        "price_to_book": raw_info.get("priceToBook"),
        "dividend_yield": raw_info.get("dividendYield"),
    }
    return history, info


def calculate_metrics(history: pd.DataFrame) -> dict[str, Any]:
    latest = history.iloc[-1]
    previous_close = history.iloc[-2]["Close"] if len(history) > 1 else latest["Close"]
    first_close = history.iloc[0]["Close"]
    return {
        "date": history.index[-1].strftime("%Y-%m-%d"),
        "close": float(latest["Close"]),
        "daily_change": float(latest["Close"] - previous_close),
        "daily_change_pct": float(
            (latest["Close"] / previous_close - 1) * 100
            if previous_close
            else 0
        ),
        "period_return_pct": float(
            (latest["Close"] / first_close - 1) * 100 if first_close else 0
        ),
        "period_high": float(history["High"].max()),
        "period_low": float(history["Low"].min()),
        "avg_volume": float(history["Volume"].mean()),
    }


def format_optional_number(value: Any, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "데이터 없음"
    return f"{float(value):,.2f}{suffix}"


def build_stock_context(
    symbol: str,
    info: dict[str, Any],
    history: pd.DataFrame,
    metrics: dict[str, Any],
) -> str:
    recent = history.tail(15).copy()
    recent.index = recent.index.strftime("%Y-%m-%d")
    recent = recent.round(2)

    market_cap = info.get("market_cap")
    market_cap_text = (
        f"{market_cap:,.0f} {info['currency']}" if market_cap else "데이터 없음"
    )
    dividend_yield = info.get("dividend_yield")
    dividend_text = (
        f"{float(dividend_yield) * 100:.2f}%"
        if dividend_yield is not None
        else "데이터 없음"
    )

    return f"""
[현재 선택 종목]
- 종목명: {info['name']}
- Yahoo Finance 심볼: {symbol}
- 거래소: {info['exchange'] or '데이터 없음'}
- 통화: {info['currency']}
- 업종: {info['sector'] or '데이터 없음'} / {info['industry'] or '데이터 없음'}

[조회 기간 핵심 지표]
- 데이터 기준일: {metrics['date']}
- 종가: {metrics['close']:,.0f} {info['currency']}
- 전일 대비: {metrics['daily_change']:+,.0f} ({metrics['daily_change_pct']:+.2f}%)
- 조회 기간 수익률: {metrics['period_return_pct']:+.2f}%
- 조회 기간 최고가/최저가: {metrics['period_high']:,.0f} / {metrics['period_low']:,.0f}
- 조회 기간 평균 거래량: {metrics['avg_volume']:,.0f}
- 시가총액: {market_cap_text}
- PER: {format_optional_number(info.get('trailing_pe'))}
- PBR: {format_optional_number(info.get('price_to_book'))}
- 배당수익률: {dividend_text}

[최근 15거래일 OHLCV]
{recent.to_csv(index_label='Date')}
""".strip()


def stream_answer(
    client: OpenAI,
    messages: list[dict[str, str]],
    stock_context: str,
) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"{BASE_SYSTEM_PROMPT}\n\n{stock_context}",
            },
            *messages,
        ],
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text


if "messages" not in st.session_state:
    st.session_state.messages = []
if "stock_config" not in st.session_state:
    st.session_state.stock_config = {
        "code": "005930",
        "market": "KOSPI",
        "period_label": "3개월",
        "symbol": "005930.KS",
    }

st.title("📈 AI 국내 주식 분석 챗봇")
st.markdown(
    '<p class="app-subtitle">국내 주가 데이터를 불러오고 AI와 함께 분석해 보세요.</p>',
    unsafe_allow_html=True,
)
st.markdown(f'<span class="model-badge">{MODEL}</span>', unsafe_allow_html=True)

with st.sidebar:
    st.header("종목 조회")
    with st.form("stock_search"):
        code = st.text_input(
            "종목코드",
            value=st.session_state.stock_config["code"],
            help="한국거래소 종목코드 6자리를 입력하세요.",
        )
        market = st.selectbox(
            "시장",
            list(MARKET_SUFFIXES),
            index=list(MARKET_SUFFIXES).index(
                st.session_state.stock_config["market"]
            ),
        )
        period_label = st.selectbox(
            "조회 기간",
            list(PERIOD_LABELS),
            index=list(PERIOD_LABELS).index(
                st.session_state.stock_config["period_label"]
            ),
        )
        submitted = st.form_submit_button(
            "주가 데이터 불러오기",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            new_symbol = make_symbol(code, market)
        except ValueError as error:
            st.error(str(error))
        else:
            if new_symbol != st.session_state.stock_config["symbol"]:
                st.session_state.messages = []
            st.session_state.stock_config = {
                "code": code.strip().upper(),
                "market": market,
                "period_label": period_label,
                "symbol": new_symbol,
            }
            st.rerun()

    st.divider()
    st.caption("AI 모델")
    st.code(MODEL, language=None)
    if st.button("🗑️ 대화 내용 지우기", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("주가 데이터는 15분간 캐시된 뒤 다시 조회됩니다.")

config = st.session_state.stock_config
symbol = config["symbol"]

try:
    with st.spinner(f"{symbol} 주가 데이터를 불러오는 중입니다..."):
        history, stock_info = fetch_stock_data(
            symbol,
            PERIOD_LABELS[config["period_label"]],
        )
    metrics = calculate_metrics(history)
    stock_context = build_stock_context(symbol, stock_info, history, metrics)
except Exception as error:
    st.error(f"주가 데이터를 불러오지 못했습니다: {error}")
    st.info("종목코드와 KOSPI/KOSDAQ 시장 선택이 맞는지 확인해 주세요.")
    st.stop()

st.subheader(f"{stock_info['name']} · {symbol}")
metric_columns = st.columns(4)
metric_columns[0].metric(
    "최근 종가",
    f"{metrics['close']:,.0f} {stock_info['currency']}",
    f"{metrics['daily_change_pct']:+.2f}%",
)
metric_columns[1].metric(
    f"{config['period_label']} 수익률",
    f"{metrics['period_return_pct']:+.2f}%",
)
metric_columns[2].metric("기간 최고가", f"{metrics['period_high']:,.0f}")
metric_columns[3].metric("평균 거래량", f"{metrics['avg_volume']:,.0f}")
st.caption(
    f"데이터 기준일: {metrics['date']} · 출처: Yahoo Finance(yfinance) · "
    "실시간 시세가 아니며 지연 또는 오류가 있을 수 있습니다."
)

chart_tab, data_tab, chat_tab = st.tabs(["📊 주가 차트", "🧾 최근 데이터", "💬 AI 분석"])

with chart_tab:
    st.line_chart(history["Close"], height=380, color="#10A37F")

with data_tab:
    display_data = history.tail(20).sort_index(ascending=False).copy()
    display_data.index = display_data.index.strftime("%Y-%m-%d")
    display_data.columns = ["시가", "고가", "저가", "종가", "거래량"]
    st.dataframe(
        display_data.style.format(
            {
                "시가": "{:,.0f}",
                "고가": "{:,.0f}",
                "저가": "{:,.0f}",
                "종가": "{:,.0f}",
                "거래량": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )

with chat_tab:
    st.markdown(
        '<div class="notice">AI 답변은 정보 제공용이며 투자 자문이 아닙니다. '
        "중요한 투자 결정 전에는 공시와 공식 시세를 별도로 확인하세요.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    api_key = get_api_key()
    if not api_key:
        st.warning(
            "OpenAI API 키가 설정되지 않았습니다. Streamlit Community Cloud의 "
            '**App settings → Secrets**에 `OPENAI_API_KEY = "sk-..."` 형식으로 '
            "등록하면 AI 분석 기능이 활성화됩니다."
        )

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                f"현재 **{stock_info['name']}({symbol})** 데이터를 불러왔습니다. "
                "기간 수익률, 변동, 거래량 등에 대해 질문해 보세요."
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        f"{stock_info['name']} 데이터에 대해 질문하세요",
        disabled=not bool(api_key),
    )
    if prompt and api_key:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                client = OpenAI(api_key=api_key)
                answer = st.write_stream(
                    stream_answer(
                        client,
                        st.session_state.messages,
                        stock_context,
                    )
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
