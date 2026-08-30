"""
Unit test cho app/core/llm.py.

Test 2 phần logic thuần, không cần gọi API thật:
1. _extract_text: chuẩn hoá nhiều dạng response.content khác nhau về string
2. rewrite_query: parse output của LLM thành list câu hỏi, và không được
   làm crash pipeline chính khi LLM lỗi.

generate_answer (lời gọi API thật) được mock hoàn toàn.
"""

from unittest.mock import patch

import pytest

from app.core import llm


# ---------- _extract_text ----------

def test_extract_text_from_plain_string():
    assert llm._extract_text("câu trả lời đơn giản") == "câu trả lời đơn giản"


def test_extract_text_from_list_of_text_blocks():
    content = [
        {"type": "text", "text": "Phần 1. "},
        {"type": "text", "text": "Phần 2."},
    ]
    assert llm._extract_text(content) == "Phần 1. Phần 2."


def test_extract_text_ignores_non_text_blocks():
    content = [
        {"type": "text", "text": "Câu trả lời thật. "},
        {"type": "thought_signature", "signature": "abc123"},
    ]
    assert llm._extract_text(content) == "Câu trả lời thật. "


def test_extract_text_from_list_of_plain_strings():
    content = ["Đoạn A. ", "Đoạn B."]
    assert llm._extract_text(content) == "Đoạn A. Đoạn B."


def test_extract_text_from_unexpected_type_falls_back_to_str():
    assert llm._extract_text(12345) == "12345"


# ---------- generate_answer ----------

def test_generate_answer_raises_on_empty_prompt():
    with pytest.raises(ValueError):
        llm.generate_answer("   ")


# ---------- rewrite_query ----------

def test_rewrite_query_empty_question_returns_as_is():
    assert llm.rewrite_query("") == [""]


def test_rewrite_query_parses_multiple_lines():
    fake_output = "Câu hỏi biến thể 1\nCâu hỏi biến thể 2\nCâu hỏi biến thể 3"

    with patch.object(llm, "generate_answer", return_value=fake_output):
        result = llm.rewrite_query("câu hỏi gốc", num_variants=3)

    assert result[0] == "câu hỏi gốc"  # câu gốc luôn ở vị trí đầu
    assert "Câu hỏi biến thể 1" in result
    assert "Câu hỏi biến thể 2" in result
    assert "Câu hỏi biến thể 3" in result
    assert len(result) == 4


def test_rewrite_query_strips_numbering_and_bullets():
    fake_output = "1. Biến thể một\n- Biến thể hai\n2) Biến thể ba"

    with patch.object(llm, "generate_answer", return_value=fake_output):
        result = llm.rewrite_query("gốc", num_variants=3)

    assert "Biến thể một" in result
    assert "Biến thể hai" in result
    assert "Biến thể ba" in result
    # Không được còn sót số thứ tự hay dấu gạch đầu dòng
    assert not any(r.startswith(("1.", "-", "2)")) for r in result)


def test_rewrite_query_dedupes_case_insensitive():
    fake_output = "Biến thể A\nbiến thể a\nBiến Thể A"

    with patch.object(llm, "generate_answer", return_value=fake_output):
        result = llm.rewrite_query("gốc", num_variants=3)

    # Chỉ giữ 1 bản duy nhất của "biến thể a" (không phân biệt hoa/thường)
    lowered = [r.lower() for r in result]
    assert lowered.count("biến thể a") == 1


def test_rewrite_query_falls_back_to_original_on_llm_error():
    with patch.object(llm, "generate_answer", side_effect=RuntimeError("hết quota")):
        result = llm.rewrite_query("câu hỏi gốc", num_variants=3)

    assert result == ["câu hỏi gốc"]


def test_rewrite_query_filters_preamble_lines():
    fake_output = "Dưới đây là 3 câu hỏi:\nBiến thể thật số 1\nBiến thể thật số 2"

    with patch.object(llm, "generate_answer", return_value=fake_output):
        result = llm.rewrite_query("gốc", num_variants=3)

    assert not any(r.lower().startswith("dưới đây") for r in result)
    assert "Biến thể thật số 1" in result