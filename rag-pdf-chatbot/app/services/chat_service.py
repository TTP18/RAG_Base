"""
Service xử lý việc hỏi đáp (chat) với tài liệu đã nạp.
Điều phối: gọi qa_chain, format kết quả cho UI.

Nâng cấp: truyền tham số lọc tài liệu (sources) và bật/tắt query rewriting
xuống qa_chain.
"""

import logging

from app.chains.qa_chain import ask
from app.core.vectorstore import list_sources

logger = logging.getLogger(__name__)


def answer_question(
    question: str,
    top_k: int = 4,
    sources: list[str] | None = None,
    use_query_rewriting: bool = False,
) -> dict:
    """
    Xử lý 1 câu hỏi từ user, trả về câu trả lời kèm nguồn tham khảo.

    Args:
        question: câu hỏi của user
        top_k: số đoạn context dùng để trả lời
        sources: danh sách tên tài liệu để giới hạn phạm vi tìm kiếm.
                 None hoặc rỗng = tìm trên toàn bộ tài liệu.
        use_query_rewriting: bật tính năng viết lại câu hỏi để mở rộng
                              phạm vi tìm kiếm ngữ nghĩa.

    Returns:
        dict gồm:
            - answer: câu trả lời
            - sources: danh sách đoạn text tham khảo (đã rút gọn để hiển thị UI)
            - has_answer: bool, False nếu không tìm được tài liệu liên quan
            - queries_used: các câu hỏi thực tế đã dùng để search (debug/hiển thị)
    """
    question = question.strip()

    if not question:
        return {
            "answer": "Vui lòng nhập câu hỏi.",
            "sources": [],
            "has_answer": False,
            "queries_used": [],
        }

    try:
        result = ask(
            question,
            top_k=top_k,
            sources=sources,
            use_query_rewriting=use_query_rewriting,
        )
    except Exception as e:
        # Không để lỗi (hết quota, timeout, lỗi mạng...) làm crash UI.
        # Log lại chi tiết để debug, còn user chỉ thấy thông báo thân thiện.
        logger.error(f"Lỗi khi xử lý câu hỏi: {e}")
        return {
            "answer": (
                "Xin lỗi, hệ thống gặp sự cố khi xử lý câu hỏi này "
                "(có thể do quá tải hoặc lỗi kết nối tới dịch vụ AI). "
                "Vui lòng thử lại sau ít phút."
            ),
            "sources": [],
            "has_answer": False,
            "queries_used": [question],
            "error": True,
        }

    return {
        "answer": result["answer"],
        "sources": [_truncate(src) for src in result["sources"]],
        "has_answer": len(result["sources"]) > 0,
        "queries_used": result.get("queries_used", [question]),
        "error": False,
    }


def list_available_sources() -> list[str]:
    """
    Trả về danh sách tên tài liệu hiện có trong hệ thống, dùng để hiển thị
    dropdown lọc tài liệu trên UI.
    """
    return list_sources()


def _truncate(text: str, max_length: int = 300) -> str:
    """Rút gọn đoạn text dài để hiển thị gọn trong UI (kèm dấu ...)."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."