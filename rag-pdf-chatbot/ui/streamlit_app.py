"""
Giao diện Streamlit cho RAG PDF Chatbot.
File này CHỈ xử lý UI, mọi logic thật đều gọi từ app/services/.

Thiết kế: "Thư viện số" (Digital Archive) — lấy cảm hứng từ không gian đọc
tài liệu học thuật: nền giấy ấm, mực xanh đậm, điểm nhấn đồng cổ (brass).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import markdown as md

# Set page config TRƯỚC khi import các module app/ — vì việc import
# app/services/... sẽ kéo theo import app/config.py, nơi có thể raise
# ValueError ngay lập tức nếu thiếu GOOGLE_API_KEY. Nếu set_page_config
# chưa chạy, Streamlit sẽ hiện traceback thô thay vì giao diện lỗi đẹp.
st.set_page_config(
    page_title="Thư viện AI · RAG PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from app.services.ingest_service import ingest_pdf, save_uploaded_file, delete_document
    from app.services.chat_service import answer_question, list_available_sources
except ValueError as e:
    # Trường hợp phổ biến nhất: thiếu GOOGLE_API_KEY trong file .env
    st.error(
        "⚠️ **Chưa cấu hình được hệ thống**\n\n"
        f"{e}\n\n"
        "**Cách khắc phục:**\n"
        "1. Tạo file `.env` ở thư mục gốc project (nếu chưa có)\n"
        "2. Thêm dòng: `GOOGLE_API_KEY=your_actual_api_key_here`\n"
        "3. Lấy API key miễn phí tại "
        "[Google AI Studio](https://aistudio.google.com/app/apikey)\n"
        "4. Khởi động lại ứng dụng"
    )
    st.stop()
except Exception as e:
    st.error(f"⚠️ Không thể khởi động ứng dụng do lỗi cấu hình: {e}")
    st.stop()


def render_markdown(text: str) -> str:
    """
    Convert nội dung markdown (từ câu trả lời LLM, VD **bold**, *italic*)
    thành HTML thật để hiển thị đúng bên trong các thẻ div tùy chỉnh.

    Lý do cần hàm này: khi dùng st.markdown(..., unsafe_allow_html=True) để
    chèn HTML tùy chỉnh (card, div...), Streamlit không tự động parse cú
    pháp markdown bên trong nội dung đó nữa — nó chỉ hiển thị y nguyên
    ký tự như "**text**" thay vì in đậm. Nên phải tự convert trước.
    """
    if not text:
        return ""
    return md.markdown(text, extensions=["nl2br"])


# ============================================================
# THEME TOKENS — Digital Archive
# ============================================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,500;0,700;1,500&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --paper: #F6F1E7;
        --paper-raised: #FBF8F1;
        --ink: #1B2430;
        --ink-soft: #3D4759;
        --ink-faint: #7A8296;
        --brass: #B8873D;
        --brass-soft: #E7D2A8;
        --line: #DDD4C0;
    }

    .stApp { background: var(--paper); }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--ink);
    }

    h1, h2, h3 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        color: var(--ink) !important;
        letter-spacing: -0.01em;
    }

    .archive-masthead {
        border-bottom: 2px solid var(--ink);
        padding-bottom: 1.1rem;
        margin-bottom: 0.3rem;
    }
    .archive-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--brass);
        margin-bottom: 0.3rem;
    }
    .archive-title {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--ink);
        line-height: 1.15;
        margin: 0;
    }
    .archive-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        color: var(--ink-soft);
        margin-top: 0.35rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--paper-raised);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-size: 1.05rem !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: var(--ink-soft);
        font-size: 0.88rem;
    }

    .sidebar-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--brass);
        margin-bottom: 0.5rem;
        margin-top: 0.2rem;
    }

    [data-testid="stFileUploader"] {
        border: 1.5px dashed var(--line);
        border-radius: 6px;
        padding: 0.6rem;
        background: var(--paper);
    }

    .stButton > button {
        background: var(--ink) !important;
        color: var(--paper) !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
        padding: 0.5rem 1rem !important;
        transition: background 0.15s ease;
    }
    .stButton > button:hover {
        background: var(--brass) !important;
        color: var(--ink) !important;
    }

    .doc-chip {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 5px;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.4rem;
        font-size: 0.83rem;
        color: var(--ink-soft);
    }
    .doc-chip .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--brass);
        flex-shrink: 0;
    }

    hr {
        border-color: var(--line) !important;
        margin: 1.1rem 0 !important;
    }

    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.3rem 0;
    }

    .assistant-card {
        background: var(--paper-raised);
        border: 1px solid var(--line);
        border-left: 3px solid var(--brass);
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 0.2rem 0;
    }
    .assistant-card p { margin: 0 0 0.6rem 0; }
    .assistant-card p:last-child { margin-bottom: 0; }
    .assistant-card ul, .assistant-card ol { margin: 0.4rem 0; padding-left: 1.4rem; }
    .assistant-card strong { color: var(--ink); font-weight: 700; }
    .assistant-card code {
        background: var(--brass-soft);
        padding: 0.1rem 0.35rem;
        border-radius: 3px;
        font-size: 0.85em;
    }

    .assistant-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--brass);
        margin-bottom: 0.5rem;
    }

    .user-card {
        background: var(--ink);
        color: var(--paper);
        border-radius: 6px;
        padding: 0.7rem 1.1rem;
        max-width: 75%;
        margin-left: auto;
        font-size: 0.95rem;
    }

    .source-item {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: 5px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        color: var(--ink-soft);
        line-height: 1.5;
        font-family: 'Source Serif 4', Georgia, serif;
    }
    .source-tag {
        display: inline-block;
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--brass);
        background: var(--brass-soft);
        border-radius: 3px;
        padding: 0.1rem 0.4rem;
        margin-bottom: 0.4rem;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
        background: var(--paper) !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.83rem !important;
        font-weight: 600 !important;
        color: var(--ink-soft) !important;
    }

    [data-testid="stChatInput"] {
        border: 1.5px solid var(--ink) !important;
        border-radius: 8px !important;
        background: var(--paper-raised) !important;
    }

    .empty-state {
        text-align: center;
        padding: 3.5rem 1rem;
        color: var(--ink-faint);
    }
    .empty-state .icon {
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
        opacity: 0.6;
    }
    .empty-state .title {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.2rem;
        color: var(--ink-soft);
        margin-bottom: 0.3rem;
    }
    .empty-state .desc {
        font-size: 0.85rem;
        max-width: 340px;
        margin: 0 auto;
    }

    .stSlider label {
        color: var(--ink-soft) !important;
        font-size: 0.83rem !important;
    }

    [data-testid="stAlert"] {
        border-radius: 6px !important;
        font-size: 0.85rem !important;
    }

    #MainMenu, header[data-testid="stHeader"] { background: transparent; }
    footer { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# MASTHEAD
# ============================================================
st.markdown(
    """
    <div class="archive-masthead">
        <div class="archive-eyebrow">Retrieval-Augmented Generation</div>
        <div class="archive-title">📚 Thư viện AI</div>
        <div class="archive-subtitle">
            Tra cứu tài liệu bằng ngôn ngữ tự nhiên — mọi câu trả lời đều bám sát nguồn gốc.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-label">Nạp tài liệu</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Chọn file PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("📥  Nạp vào hệ thống", type="primary", use_container_width=True):
            with st.spinner("Đang đọc, chia nhỏ và mã hoá tài liệu…"):
                try:
                    file_bytes = uploaded_file.read()
                    saved_path = save_uploaded_file(file_bytes, uploaded_file.name)
                    result = ingest_pdf(saved_path)

                    if result["status"] == "success":
                        st.success(
                            f"Đã nạp {result['num_chunks']} đoạn · "
                            f"{result['num_characters']:,} ký tự"
                        )
                        st.rerun()
                    elif result["status"] == "duplicate":
                        st.warning(result["message"])
                    else:  # status == "error"
                        st.error(result["message"])
                except Exception as e:
                    # Lỗi ngoài dự kiến (VD: không lưu được file lên đĩa)
                    # — ingest_pdf đã tự bắt lỗi nội bộ, đây chỉ là lớp bảo hiểm cuối.
                    st.error(f"Lỗi khi nạp tài liệu: {e}")

    st.divider()

    st.markdown('<div class="sidebar-label">Tài liệu trong hệ thống</div>', unsafe_allow_html=True)

    # Gọi 1 lần duy nhất, dùng chung cho cả danh sách quản lý lẫn dropdown lọc bên dưới
    available_sources = list_available_sources()

    if available_sources:
        for i, fname in enumerate(available_sources):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f'<div class="doc-chip"><span class="dot"></span>{fname}</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑️", key=f"del_{i}_{fname}", help=f"Xóa {fname}"):
                    with st.spinner(f"Đang xóa {fname}…"):
                        try:
                            result = delete_document(fname)
                            if result["status"] == "success":
                                st.success(f"Đã xóa {fname}")
                                st.rerun()
                            else:
                                # status == "partial": đã xóa vector nhưng
                                # KHÔNG xóa được file gốc — không rerun ngay
                                # để người dùng kịp đọc thông báo chi tiết.
                                st.warning(result["message"])
                        except Exception as e:
                            # Lớp bảo hiểm cuối — delete_document đã tự bắt
                            # lỗi xóa file, đây chỉ còn bắt lỗi ngoài dự
                            # kiến khác (VD: lỗi kết nối tới ChromaDB).
                            st.error(f"Lỗi khi xóa: {e}")
    else:
        st.markdown(
            '<p style="font-size:0.82rem; color:var(--ink-faint);">'
            'Chưa có tài liệu nào trong hệ thống.</p>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown('<div class="sidebar-label">Phạm vi tìm kiếm</div>', unsafe_allow_html=True)

    if available_sources:
        selected_sources = st.multiselect(
            "Chỉ tìm trong tài liệu (bỏ trống = tìm tất cả)",
            options=available_sources,
            default=[],
        )
    else:
        selected_sources = []
        st.markdown(
            '<p style="font-size:0.8rem; color:var(--ink-faint);">'
            'Chưa có tài liệu nào trong hệ thống.</p>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-label">Tuỳ chỉnh</div>', unsafe_allow_html=True)
    top_k = st.slider("Số đoạn ngữ cảnh dùng để trả lời", min_value=2, max_value=10, value=4)

    use_query_rewriting = st.toggle(
        "Mở rộng câu hỏi (query rewriting)",
        value=False,
        help="Sinh thêm các cách diễn đạt khác của câu hỏi để tăng khả năng "
             "tìm đúng ngữ cảnh liên quan. Chậm hơn một chút nhưng chính xác hơn "
             "với câu hỏi mà tài liệu dùng từ ngữ khác."
    )

    st.divider()
    st.markdown(
        '<p style="font-size:0.75rem; color:var(--ink-faint); line-height:1.6;">'
        'Hệ thống chỉ trả lời dựa trên nội dung tài liệu đã nạp. '
        'Nếu không tìm thấy thông tin liên quan, câu trả lời sẽ nêu rõ điều đó '
        'thay vì suy đoán.</p>',
        unsafe_allow_html=True,
    )


# ============================================================
# KHU VỰC CHAT
# ============================================================

def render_message(message: dict, is_live: bool = False):
    """Render 1 message (user hoặc assistant) ra giao diện."""
    if message["role"] == "user":
        st.markdown(f'<div class="user-card">{message["content"]}</div>', unsafe_allow_html=True)
        return

    answer_html = render_markdown(message["content"])
    st.markdown(
        f"""
        <div class="assistant-card">
            <div class="assistant-label">Trả lời</div>
            {answer_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    sources = message.get("sources") or []
    if sources:
        with st.expander(f"📎 {len(sources)} nguồn tham khảo"):
            for i, src in enumerate(sources):
                st.markdown(
                    f"""
                    <div class="source-item">
                        <span class="source-tag">Nguồn {i+1}</span><br>
                        {src}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    queries_used = message.get("queries_used") or []
    if len(queries_used) > 1:
        with st.expander("🔄 Các cách diễn đạt đã thử"):
            for q in queries_used:
                st.markdown(f"- {q}")


if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">🗂️</div>
            <div class="title">Chưa có câu hỏi nào</div>
            <div class="desc">Nạp một tài liệu PDF ở thanh bên trái, sau đó đặt câu hỏi
            về nội dung của tài liệu đó tại ô bên dưới.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Đặt câu hỏi về tài liệu của bạn…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f'<div class="user-card">{question}</div>', unsafe_allow_html=True)

    with st.spinner("Đang tra cứu và tổng hợp câu trả lời…"):
        try:
            result = answer_question(
                question,
                top_k=top_k,
                sources=selected_sources if selected_sources else None,
                use_query_rewriting=use_query_rewriting,
            )

            new_message = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "queries_used": result.get("queries_used", []),
            }
            st.session_state.messages.append(new_message)
            render_message(new_message)

        except Exception as e:
            error_msg = f"Đã xảy ra lỗi: {e}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "queries_used": [],
            })