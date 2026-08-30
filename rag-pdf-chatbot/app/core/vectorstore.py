"""
Module quản lý ChromaDB - nơi lưu trữ embeddings và tìm kiếm ngữ nghĩa.
Chỉ làm việc với vector store, không biết gì về PDF hay LLM.

Nâng cấp:
- Hỗ trợ lọc theo nguồn tài liệu cụ thể (metadata filter) và liệt kê
  danh sách các tài liệu đã có trong hệ thống.
- Cache lại instance Chroma bằng lru_cache — trước đây mỗi lần add/query/
  delete đều mở lại kết nối tới persist_directory từ đầu. Vì đây vẫn là
  CÙNG 1 collection trên đĩa, việc mở lại liên tục vừa chậm vừa không
  cần thiết; chỉ cần 1 instance dùng chung cho toàn bộ process.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import VECTORSTORE_DIR, COLLECTION_NAME, TOP_K_RESULTS
from app.core.embedder import get_embedder


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Khởi tạo (hoặc load lại nếu đã tồn tại) ChromaDB. Kết quả được cache
    lại (lru_cache) vì persist_directory + collection_name không đổi
    trong suốt vòng đời process — không có lý do gì để mở lại kết nối
    mỗi lần add_chunks/search_similar/delete_source được gọi.

    Returns:
        Instance Chroma, sẵn sàng add/query (dùng chung, cached)
    """
    embedder = get_embedder()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=str(VECTORSTORE_DIR),
    )
    return vectorstore


def add_chunks(chunks: list[str], source_name: str) -> None:
    """
    Thêm danh sách các đoạn text vào vector store.

    Args:
        chunks: danh sách các đoạn text (từ splitter.py)
        source_name: tên file nguồn, lưu vào metadata để biết chunk
                     này thuộc file nào (hữu ích khi có nhiều PDF)
    """
    if not chunks:
        raise ValueError("Danh sách chunks rỗng, không có gì để thêm")

    vectorstore = get_vectorstore()

    documents = [
        Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]

    vectorstore.add_documents(documents)


def search_similar(
    query: str,
    top_k: int = TOP_K_RESULTS,
    sources: list[str] | None = None,
) -> list[Document]:
    """
    Tìm các đoạn văn bản liên quan nhất tới câu hỏi.

    Args:
        query: câu hỏi hoặc câu truy vấn của user
        top_k: số lượng kết quả trả về
        sources: danh sách tên tài liệu để giới hạn phạm vi tìm kiếm.
                 None hoặc rỗng = tìm trên toàn bộ tài liệu đã nạp.

    Returns:
        Danh sách các Document liên quan nhất, sắp xếp theo độ liên quan giảm dần
    """
    vectorstore = get_vectorstore()

    filter_dict = None
    if sources:
        if len(sources) == 1:
            filter_dict = {"source": sources[0]}
        else:
            filter_dict = {"source": {"$in": sources}}

    results = vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
    return results


def search_similar_with_score(
    query: str,
    top_k: int = TOP_K_RESULTS,
    sources: list[str] | None = None,
) -> list[tuple[Document, float]]:
    """
    Giống search_similar, nhưng trả kèm điểm khoảng cách (distance score)
    của từng kết quả. Dùng khi cần GỘP kết quả từ NHIỀU câu hỏi khác nhau
    (VD: query rewriting) rồi rerank lại theo độ liên quan thật, thay vì
    chỉ nối danh sách theo thứ tự các câu hỏi.

    Lưu ý: Chroma trả về "distance" (càng NHỎ càng liên quan), không phải
    "similarity" (càng LỚN càng liên quan) — cần để ý khi sort.

    Args:
        query: câu hỏi hoặc câu truy vấn của user
        top_k: số lượng kết quả trả về
        sources: danh sách tên tài liệu để giới hạn phạm vi tìm kiếm

    Returns:
        Danh sách (Document, distance_score), sắp xếp theo distance tăng dần
        (kết quả đầu tiên là liên quan nhất)
    """
    vectorstore = get_vectorstore()

    filter_dict = None
    if sources:
        if len(sources) == 1:
            filter_dict = {"source": sources[0]}
        else:
            filter_dict = {"source": {"$in": sources}}

    results = vectorstore.similarity_search_with_score(query, k=top_k, filter=filter_dict)
    return results


def list_sources() -> list[str]:
    """
    Liệt kê danh sách tên các tài liệu (nguồn) đã có trong vector store.
    Dùng để hiển thị dropdown lọc tài liệu trên UI.

    Returns:
        Danh sách tên file duy nhất, đã sắp xếp
    """
    vectorstore = get_vectorstore()

    try:
        all_data = vectorstore.get()
    except Exception:
        return []

    metadatas = all_data.get("metadatas", []) or []
    sources = {meta["source"] for meta in metadatas if meta and "source" in meta}
    return sorted(sources)


def delete_source(source_name: str) -> int:
    """
    Xóa toàn bộ chunks thuộc về 1 tài liệu cụ thể khỏi vector store.

    Args:
        source_name: tên tài liệu cần xóa

    Returns:
        Số lượng chunks đã bị xóa
    """
    vectorstore = get_vectorstore()

    existing = vectorstore.get(where={"source": source_name})
    ids_to_delete = existing.get("ids", [])

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)

    return len(ids_to_delete)