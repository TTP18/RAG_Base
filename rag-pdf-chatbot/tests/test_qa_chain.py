"""
Unit test cho app/chains/qa_chain.py.

Phần quan trọng nhất cần test là _search_multi_query: logic gộp kết quả
từ nhiều câu hỏi (query rewriting), loại trùng lặp, và RERANK theo điểm
liên quan thật thay vì chỉ nối theo thứ tự query. Đây chính là bug đã
được sửa (xem docstring gốc trong qa_chain.py), nên cần test để đảm bảo
không bị regress về hành vi cũ.

Toàn bộ test dùng mock cho search_similar_with_score và generate_answer
— không gọi Chroma hay Gemini thật, chạy nhanh và không tốn quota.
"""

from unittest.mock import patch

from langchain_core.documents import Document

from app.chains import qa_chain


def _doc(content: str) -> Document:
    return Document(page_content=content, metadata={"source": "test.pdf"})


def test_search_multi_query_reranks_by_score_not_query_order():
    """
    Bug cũ: kết quả của câu hỏi ĐẦU TIÊN luôn được ưu tiên giữ lại dù
    không liên quan bằng kết quả của câu hỏi sau. Test này giả lập đúng
    tình huống đó: câu hỏi đầu trả về đoạn có score TỆ, câu hỏi thứ 2
    trả về đoạn có score TỐT hơn nhiều — kết quả cuối phải ưu tiên đoạn
    có score tốt hơn, bất kể nó đến từ query nào.
    """
    doc_bad = _doc("Đoạn ít liên quan")
    doc_good = _doc("Đoạn rất liên quan")

    def fake_search(query, top_k, sources):
        if query == "câu hỏi gốc":
            return [(doc_bad, 0.9)]  # distance lớn = kém liên quan
        else:
            return [(doc_good, 0.1)]  # distance nhỏ = liên quan nhất

    with patch.object(qa_chain, "search_similar_with_score", side_effect=fake_search):
        result = qa_chain._search_multi_query(
            queries=["câu hỏi gốc", "biến thể"],
            top_k=1,
            sources=None,
        )

    assert len(result) == 1
    assert result[0].page_content == "Đoạn rất liên quan"


def test_search_multi_query_dedupes_identical_content():
    doc = _doc("Nội dung trùng lặp giữa 2 query")

    def fake_search(query, top_k, sources):
        return [(doc, 0.2)]

    with patch.object(qa_chain, "search_similar_with_score", side_effect=fake_search):
        result = qa_chain._search_multi_query(
            queries=["câu hỏi A", "câu hỏi B"],
            top_k=5,
            sources=None,
        )

    assert len(result) == 1


def test_search_multi_query_respects_top_k_limit():
    docs = [(_doc(f"đoạn {i}"), float(i)) for i in range(10)]

    def fake_search(query, top_k, sources):
        return docs

    with patch.object(qa_chain, "search_similar_with_score", side_effect=fake_search):
        result = qa_chain._search_multi_query(
            queries=["chỉ 1 câu hỏi"],
            top_k=3,
            sources=None,
        )

    assert len(result) == 3
    # Phải là 3 đoạn có distance NHỎ nhất (liên quan nhất)
    assert [d.page_content for d in result] == ["đoạn 0", "đoạn 1", "đoạn 2"]


def test_search_multi_query_empty_when_no_candidates():
    with patch.object(qa_chain, "search_similar_with_score", return_value=[]):
        result = qa_chain._search_multi_query(
            queries=["câu hỏi không tìm thấy gì"],
            top_k=4,
            sources=None,
        )

    assert result == []


def test_ask_raises_on_empty_question():
    import pytest

    with pytest.raises(ValueError):
        qa_chain.ask("   ")


def test_ask_returns_fallback_message_when_no_docs_found():
    with patch.object(qa_chain, "_search_multi_query", return_value=[]):
        result = qa_chain.ask("câu hỏi bất kỳ", use_query_rewriting=False)

    assert result["sources"] == []
    assert "Không tìm thấy" in result["answer"]
    assert result["queries_used"] == ["câu hỏi bất kỳ"]


def test_ask_builds_prompt_with_original_question_not_variants():
    """
    Theo thiết kế: dù bật query rewriting, prompt gửi cho LLM phải luôn
    dùng câu hỏi GỐC (không dùng biến thể), để câu trả lời bám sát đúng
    ý người dùng hỏi thật sự.
    """
    doc = _doc("ngữ cảnh liên quan")

    with patch.object(qa_chain, "rewrite_query", return_value=["câu hỏi gốc", "biến thể X"]), \
         patch.object(qa_chain, "_search_multi_query", return_value=[doc]), \
         patch.object(qa_chain, "generate_answer") as mock_generate:

        mock_generate.return_value = "câu trả lời giả lập"

        result = qa_chain.ask(
            "câu hỏi gốc",
            use_query_rewriting=True,
        )

        sent_prompt = mock_generate.call_args[0][0]
        assert "câu hỏi gốc" in sent_prompt
        assert "biến thể X" not in sent_prompt

    assert result["answer"] == "câu trả lời giả lập"
    assert result["queries_used"] == ["câu hỏi gốc", "biến thể X"]