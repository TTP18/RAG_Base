"""
Pipeline RAG hoàn chỉnh: nhận câu hỏi -> retrieve context -> generate answer.
Đây là nơi ghép nối core/vectorstore.py và core/llm.py lại với nhau.

Nâng cấp:
- Hỗ trợ lọc theo tài liệu cụ thể (multi-document filtering)
- Hỗ trợ query rewriting: sinh nhiều biến thể câu hỏi để tăng khả năng
  tìm đúng ngữ cảnh liên quan, đặc biệt hữu ích khi câu hỏi dùng từ ngữ
  khác với cách diễn đạt trong tài liệu gốc.
"""

from app.core.vectorstore import search_similar_with_score
from app.core.llm import generate_answer, rewrite_query


PROMPT_TEMPLATE = """Bạn là trợ lý AI trả lời câu hỏi dựa trên tài liệu được cung cấp.
Chỉ trả lời dựa trên thông tin trong phần "Ngữ cảnh" bên dưới.
Nếu ngữ cảnh không đủ thông tin để trả lời, hãy nói rõ là không tìm thấy thông tin liên quan trong tài liệu, không được bịa ra câu trả lời.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""


def ask(
    question: str,
    top_k: int = 4,
    sources: list[str] | None = None,
    use_query_rewriting: bool = False,
) -> dict:
    """
    Chạy toàn bộ pipeline RAG cho 1 câu hỏi.

    Args:
        question: câu hỏi của user
        top_k: số đoạn context lấy ra để đưa vào prompt (áp dụng sau khi
               đã gộp kết quả từ mọi biến thể câu hỏi, nếu có rewriting)
        sources: danh sách tên tài liệu để giới hạn phạm vi tìm kiếm.
                 None = tìm trên toàn bộ tài liệu đã nạp.
        use_query_rewriting: nếu True, sinh thêm các biến thể của câu hỏi
                              để mở rộng phạm vi tìm kiếm ngữ nghĩa.

    Returns:
        dict gồm:
            - answer: câu trả lời từ LLM
            - sources: danh sách các đoạn context đã dùng (để hiển thị UI)
            - queries_used: danh sách các câu hỏi thực tế đã dùng để search
                             (hữu ích để debug/hiển thị khi bật query rewriting)
    """
    if not question or not question.strip():
        raise ValueError("Câu hỏi rỗng")

    # Bước 1: xác định danh sách câu hỏi sẽ dùng để search
    if use_query_rewriting:
        queries = rewrite_query(question, num_variants=3)
    else:
        queries = [question]

    # Bước 2: search với từng câu hỏi, gộp kết quả và loại trùng lặp
    relevant_docs = _search_multi_query(queries, top_k=top_k, sources=sources)

    if not relevant_docs:
        return {
            "answer": "Không tìm thấy tài liệu nào liên quan để trả lời câu hỏi này.",
            "sources": [],
            "queries_used": queries,
        }

    # Bước 3: ghép context từ các đoạn tìm được
    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

    # Bước 4: build prompt hoàn chỉnh (luôn dùng câu hỏi GỐC, không dùng
    # biến thể, để câu trả lời bám sát đúng ý người dùng hỏi)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # Bước 5: gọi LLM sinh câu trả lời
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in relevant_docs],
        "queries_used": queries,
    }


def _search_multi_query(
    queries: list[str],
    top_k: int,
    sources: list[str] | None,
) -> list:
    """
    Chạy search cho nhiều câu hỏi (biến thể), gộp kết quả lại, loại trùng
    lặp (dựa trên nội dung đoạn văn bản), rồi RERANK theo điểm liên quan
    thật (distance score) trước khi cắt còn top_k.

    Trước đây hàm này chỉ nối kết quả của từng câu hỏi theo thứ tự rồi
    cắt [:top_k] — nghĩa là các đoạn của câu hỏi ĐẦU TIÊN luôn được ưu
    tiên giữ lại, bất kể đoạn đó có thực sự liên quan nhất hay không.
    Giờ đây mọi ứng viên (từ mọi câu hỏi) được so sánh công bằng với
    nhau bằng điểm số, đảm bảo top_k cuối cùng là top_k liên quan NHẤT
    trong toàn bộ ứng viên, không phải top_k theo thứ tự query.

    Args:
        queries: danh sách câu hỏi (câu gốc + biến thể nếu có)
        top_k: số kết quả tối đa cần giữ lại sau khi gộp
        sources: bộ lọc tài liệu (xem search_similar)

    Returns:
        Danh sách Document duy nhất, tối đa top_k phần tử, sắp xếp theo
        độ liên quan giảm dần
    """
    seen_content = set()
    candidates = []  # list of (doc, distance_score)

    # Trước đây mỗi câu hỏi (kể cả các biến thể từ query rewriting) đều
    # gọi search_similar_with_score với đúng top_k — nghĩa là bật
    # rewriting với 4 câu hỏi thì tốn 4 lượt gọi vector store, mỗi lượt
    # xin đủ top_k kết quả, dù cuối cùng cũng chỉ giữ lại top_k sau khi
    # rerank. Giờ đây, khi có nhiều hơn 1 query, mỗi query chỉ cần xin
    # đủ số ứng viên hợp lý (thay vì full top_k mỗi lần) — vẫn đủ để
    # rerank công bằng, nhưng giảm tải cho vector store.
    per_query_k = top_k if len(queries) <= 1 else max(2, -(-top_k * 2 // len(queries)))

    # Với mỗi câu hỏi, lấy per_query_k ứng viên riêng kèm điểm số — đảm
    # bảo đủ ứng viên trước khi gộp và rerank lại toàn bộ
    for q in queries:
        results = search_similar_with_score(q, top_k=per_query_k, sources=sources)
        for doc, score in results:
            key = doc.page_content.strip()
            if key not in seen_content:
                seen_content.add(key)
                candidates.append((doc, score))

    # Chroma trả về "distance" — càng NHỎ càng liên quan, nên sort tăng dần
    candidates.sort(key=lambda pair: pair[1])

    return [doc for doc, _score in candidates[:top_k]]