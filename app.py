import os
import hashlib
from io import BytesIO
from typing import Any, Iterator

import pandas as pd
import streamlit as st
import yfinance as yf
from openai import OpenAI
from pypdf import PdfReader


MODEL = "gpt-4o-mini"
PERIOD_LABELS = {
    "1개월": "1mo",
    "3개월": "3mo",
    "6개월": "6mo",
    "1년": "1y",
    "2년": "2y",
}
MARKET_SUFFIXES = {"KOSPI": ".KS", "KOSDAQ": ".KQ"}
MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_PDF_CHARS = 60_000
BASE_SYSTEM_PROMPT = """
당신은 한국 주식 데이터 분석을 돕는 친절하고 신중한 AI 어시스턴트입니다.
아래에 제공된 yfinance 데이터만을 현재 주가 데이터의 근거로 사용하세요.
PDF 문서가 제공된 경우 해당 문서의 내용만을 문서 관련 답변의 근거로 사용하세요.
PDF 근거를 사용한 문장에는 가능한 한 [페이지 N] 형식으로 페이지를 표시하세요.
문서나 데이터에서 확인할 수 없는 내용은 추측하지 말고 확인할 수 없다고 답하세요.
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


@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes, file_name: str) -> dict[str, Any]:
    """Extract page-tagged text from a PDF within safe context limits."""
    if len(file_bytes) > MAX_PDF_BYTES:
        raise ValueError("PDF 파일은 최대 15MB까지 업로드할 수 있습니다.")

    reader = PdfReader(BytesIO(file_bytes))
    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
        except Exception as error:
            raise ValueError("암호화된 PDF는 분석할 수 없습니다.") from error
        if decrypt_result == 0:
            raise ValueError("암호가 설정된 PDF는 분석할 수 없습니다.")

    total_pages = len(reader.pages)
    pages_to_read = min(total_pages, MAX_PDF_PAGES)
    parts: list[str] = []
    extracted_chars = 0
    truncated = total_pages > MAX_PDF_PAGES

    for page_number in range(1, pages_to_read + 1):
        try:
            page_text = reader.pages[page_number - 1].extract_text() or ""
        except Exception:
            page_text = ""
        page_text = " ".join(page_text.split())
        if not page_text:
            continue

        page_block = f"[페이지 {page_number}]\n{page_text}\n"
        remaining = MAX_PDF_CHARS - extracted_chars
        if remaining <= 0:
            truncated = True
            break
        if len(page_block) > remaining:
            parts.append(page_block[:remaining])
            extracted_chars += remaining
            truncated = True
            break
        parts.append(page_block)
        extracted_chars += len(page_block)

    text = "\n".join(parts).strip()
    if not text:
        raise ValueError(
            "추출할 수 있는 텍스트가 없습니다. 스캔 이미지형 PDF라면 OCR 처리가 필요합니다."
        )

    return {
        "file_name": file_name,
        "total_pages": total_pages,
        "read_pages": pages_to_read,
        "text": text,
        "characters": len(text),
        "truncated": truncated,
    }


def build_pdf_context(pdf_info: dict[str, Any] | None) -> str:
    if not pdf_info:
        return "[업로드 PDF]\n업로드된 PDF 없음"

    limit_note = (
        "분석 한도로 인해 일부 내용이 생략됨"
        if pdf_info["truncated"]
        else "전체 추출 범위 포함"
    )
    return f"""
[업로드 PDF]
- 파일명: {pdf_info['file_name']}
- 전체 페이지: {pdf_info['total_pages']}
- 읽은 페이지 범위: 최대 {pdf_info['read_pages']}페이지
- 추출 문자 수: {pdf_info['characters']:,}
- 상태: {limit_note}

[PDF 추출 텍스트 시작]
{pdf_info['text']}
[PDF 추출 텍스트 끝]
""".strip()


def stream_answer(
    client: OpenAI,
    messages: list[dict[str, str]],
    stock_context: str,
    pdf_context: str,
) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{BASE_SYSTEM_PROMPT}\n\n{stock_context}\n\n{pdf_context}"
                ),
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
if "pdf_key" not in st.session_state:
    st.session_state.pdf_key = None

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
            width="stretch",
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
    st.header("PDF 문서")
    uploaded_pdf = st.file_uploader(
        "분석할 PDF 업로드",
        type=["pdf"],
        help="텍스트형 PDF, 최대 15MB·100페이지까지 분석합니다.",
    )
    st.caption(
        "PDF의 텍스트를 추출해 AI 답변의 참고 문서로 사용합니다. "
        "업로드 파일은 GitHub 저장소에 저장되지 않습니다."
    )

    st.divider()
    st.caption("AI 모델")
    st.code(MODEL, language=None)
    if st.button("🗑️ 대화 내용 지우기", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption("주가 데이터는 15분간 캐시된 뒤 다시 조회됩니다.")

config = st.session_state.stock_config
symbol = config["symbol"]

pdf_info: dict[str, Any] | None = None
pdf_error: str | None = None
current_pdf_key: str | None = None
if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    current_pdf_key = (
        f"{uploaded_pdf.name}:{len(pdf_bytes)}:"
        f"{hashlib.sha256(pdf_bytes).hexdigest()}"
    )
    try:
        with st.spinner("PDF 텍스트를 추출하는 중입니다..."):
            pdf_info = extract_pdf_text(pdf_bytes, uploaded_pdf.name)
    except Exception as error:
        pdf_error = str(error)

if current_pdf_key != st.session_state.pdf_key:
    st.session_state.messages = []
    st.session_state.pdf_key = current_pdf_key

pdf_context = build_pdf_context(pdf_info)

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

chart_tab, data_tab, pdf_tab, chat_tab = st.tabs(
    ["📊 주가 차트", "🧾 최근 데이터", "📄 PDF 문서", "💬 AI 분석"]
)

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
        width="stretch",
    )

with pdf_tab:
    if pdf_error:
        st.error(pdf_error)
    elif pdf_info:
        status_columns = st.columns(3)
        status_columns[0].metric("파일", pdf_info["file_name"])
        status_columns[1].metric("전체 페이지", f"{pdf_info['total_pages']:,}")
        status_columns[2].metric("추출 문자", f"{pdf_info['characters']:,}")
        if pdf_info["truncated"]:
            st.warning(
                "분석 한도(최대 100페이지·60,000자)로 인해 문서 일부가 생략되었습니다."
            )
        st.success("PDF 텍스트 추출이 완료되어 AI 답변에 반영됩니다.")
        with st.expander("추출 텍스트 미리보기"):
            preview = pdf_info["text"][:5_000]
            st.text(preview + ("\n\n…" if len(pdf_info["text"]) > 5_000 else ""))
        st.caption(
            "표·도표·복잡한 레이아웃은 텍스트 추출 과정에서 원본 구조가 "
            "정확히 유지되지 않을 수 있습니다."
        )
    else:
        st.info("사이드바에서 PDF 파일을 업로드하면 문서 내용을 분석할 수 있습니다.")

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
            pdf_message = (
                f" 업로드된 **{pdf_info['file_name']}** 문서 내용도 함께 참고할 수 있습니다."
                if pdf_info
                else " PDF를 업로드하면 문서 내용에 대해서도 질문할 수 있습니다."
            )
            st.markdown(
                f"현재 **{stock_info['name']}({symbol})** 데이터를 불러왔습니다. "
                f"기간 수익률, 변동, 거래량 등에 대해 질문해 보세요.{pdf_message}"
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
                        pdf_context,
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
