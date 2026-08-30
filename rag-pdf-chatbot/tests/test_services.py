"""
Unit test cho app/services/chat_service.py.

Tập trung vào phần LOGIC ĐIỀU PHỐI của service layer: xử lý câu hỏi
rỗng, không để exception làm crash UI, và rút gọn text nguồn tham khảo
— tất cả không cần gọi qa_chain.ask thật (được mock).
"""

from unittest.mock import patch

from app.services import chat_service


def test_answer_question_empty_input_returns_prompt_message():
    result = chat_service.answer_question("   ")

    assert result["has_answer"] is False
    assert result["sources"] == []
    assert "nhập câu hỏi" in result["answer"].lower()


def test_answer_question_success_path():
    fake_result = {
        "answer": "câu trả lời",
        "sources": ["nguồn 1", "nguồn 2"],
        "queries_used": ["câu hỏi"],
    }
    with patch.object(chat_service, "ask", return_value=fake_result):
        result = chat_service.answer_question("câu hỏi")

    assert result["has_answer"] is True
    assert result["answer"] == "câu trả lời"
    assert result["error"] is False
    assert len(result["sources"]) == 2


def test_answer_question_no_sources_means_has_answer_false():
    fake_result = {"answer": "không tìm thấy", "sources": [], "queries_used": ["câu hỏi"]}
    with patch.object(chat_service, "ask", return_value=fake_result):
        result = chat_service.answer_question("câu hỏi")

    assert result["has_answer"] is False


def test_answer_question_exception_returns_friendly_error_not_crash():
    with patch.object(chat_service, "ask", side_effect=RuntimeError("hết quota API")):
        result = chat_service.answer_question("câu hỏi bất kỳ")

    assert result["error"] is True
    assert result["has_answer"] is False
    assert result["sources"] == []
    # Không được để lộ chi tiết lỗi kỹ thuật thô ra cho user
    assert "hết quota API" not in result["answer"]


def test_truncate_short_text_unchanged():
    text = "văn bản ngắn"
    assert chat_service._truncate(text, max_length=300) == text


def test_truncate_long_text_adds_ellipsis():
    text = "a" * 500
    result = chat_service._truncate(text, max_length=300)

    assert len(result) == 303  # 300 ký tự + "..."
    assert result.endswith("...")


def test_list_available_sources_delegates_to_vectorstore():
    with patch.object(chat_service, "list_sources", return_value=["a.pdf", "b.pdf"]):
        result = chat_service.list_available_sources()

    assert result == ["a.pdf", "b.pdf"]