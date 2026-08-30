"""
Unit test cho app/core/vectorstore.py.

Vì Chroma cần persist_directory + gọi embedding API thật, ta không test
trực tiếp với Chroma thật. Thay vào đó, mock get_vectorstore() để trả
về 1 fake object ghi lại các lời gọi — qua đó kiểm tra ĐÚNG LOGIC mà
code của mình chịu trách nhiệm: cách build filter_dict, cách gói
Document với metadata, cách đếm/xoá theo source.
"""

from unittest.mock import patch, MagicMock

import pytest

from app.core import vectorstore as vs


class FakeChroma:
    """Giả lập Chroma đủ để kiểm tra các lời gọi được thực hiện đúng."""

    def __init__(self):
        self.added_documents = None
        self.last_search_args = None
        self.last_search_with_score_args = None
        self.get_return = {"metadatas": []}
        self.deleted_ids = None

    def add_documents(self, documents):
        self.added_documents = documents

    def similarity_search(self, query, k, filter):
        self.last_search_args = {"query": query, "k": k, "filter": filter}
        return []

    def similarity_search_with_score(self, query, k, filter):
        self.last_search_with_score_args = {"query": query, "k": k, "filter": filter}
        return []

    def get(self, where=None):
        if where is not None:
            self._last_get_where = where
        return self.get_return

    def delete(self, ids):
        self.deleted_ids = ids


@pytest.fixture(autouse=True)
def clear_cache():
    # get_vectorstore dùng lru_cache — cần clear trước mỗi test để mock
    # ở test này không rò rỉ sang test khác.
    vs.get_vectorstore.cache_clear()
    yield
    vs.get_vectorstore.cache_clear()


def test_add_chunks_builds_documents_with_correct_metadata():
    fake = FakeChroma()
    with patch.object(vs, "get_vectorstore", return_value=fake):
        vs.add_chunks(["đoạn 1", "đoạn 2"], source_name="test.pdf")

    assert len(fake.added_documents) == 2
    assert fake.added_documents[0].page_content == "đoạn 1"
    assert fake.added_documents[0].metadata == {"source": "test.pdf", "chunk_index": 0}
    assert fake.added_documents[1].metadata == {"source": "test.pdf", "chunk_index": 1}


def test_add_chunks_raises_on_empty_list():
    with pytest.raises(ValueError):
        vs.add_chunks([], source_name="test.pdf")


def test_search_similar_no_filter_when_sources_none():
    fake = FakeChroma()
    with patch.object(vs, "get_vectorstore", return_value=fake):
        vs.search_similar("câu hỏi", top_k=4, sources=None)

    assert fake.last_search_args["filter"] is None


def test_search_similar_single_source_filter():
    fake = FakeChroma()
    with patch.object(vs, "get_vectorstore", return_value=fake):
        vs.search_similar("câu hỏi", top_k=4, sources=["a.pdf"])

    assert fake.last_search_args["filter"] == {"source": "a.pdf"}


def test_search_similar_multi_source_filter_uses_in_operator():
    fake = FakeChroma()
    with patch.object(vs, "get_vectorstore", return_value=fake):
        vs.search_similar("câu hỏi", top_k=4, sources=["a.pdf", "b.pdf"])

    assert fake.last_search_args["filter"] == {"source": {"$in": ["a.pdf", "b.pdf"]}}


def test_list_sources_dedupes_and_sorts():
    fake = FakeChroma()
    fake.get_return = {
        "metadatas": [
            {"source": "b.pdf"},
            {"source": "a.pdf"},
            {"source": "b.pdf"},
            {},  # metadata thiếu "source" — không được làm crash
            None,  # metadata None — không được làm crash
        ]
    }
    with patch.object(vs, "get_vectorstore", return_value=fake):
        result = vs.list_sources()

    assert result == ["a.pdf", "b.pdf"]


def test_list_sources_returns_empty_list_on_error():
    fake = MagicMock()
    fake.get.side_effect = RuntimeError("lỗi kết nối")
    with patch.object(vs, "get_vectorstore", return_value=fake):
        result = vs.list_sources()

    assert result == []


def test_delete_source_returns_count_and_calls_delete():
    fake = FakeChroma()
    fake.get_return = {"ids": ["id1", "id2", "id3"]}
    with patch.object(vs, "get_vectorstore", return_value=fake):
        count = vs.delete_source("test.pdf")

    assert count == 3
    assert fake.deleted_ids == ["id1", "id2", "id3"]


def test_delete_source_no_op_when_nothing_to_delete():
    fake = FakeChroma()
    fake.get_return = {"ids": []}
    with patch.object(vs, "get_vectorstore", return_value=fake):
        count = vs.delete_source("khong-ton-tai.pdf")

    assert count == 0
    assert fake.deleted_ids is None