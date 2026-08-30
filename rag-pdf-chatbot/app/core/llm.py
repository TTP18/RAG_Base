"""
Module gọi LLM (Gemini) để sinh câu trả lời.
Chỉ nhận prompt (text) -> trả về text.
Không biết gì về retrieval, vector store, hay RAG.

Nâng cấp: thêm hàm rewrite_query để sinh các biến thể của câu hỏi,
giúp cải thiện khả năng tìm kiếm ngữ nghĩa (semantic search) khi câu hỏi
gốc dùng từ ngữ không khớp với cách diễn đạt trong tài liệu.
"""

import re
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GOOGLE_API_KEY, LLM_MODEL_NAME

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Khởi tạo LLM model.

    Args:
        temperature: độ "sáng tạo" của câu trả lời (0 = chính xác/ổn định,
                     1 = sáng tạo/ngẫu nhiên hơn). Với RAG nên để thấp
                     để câu trả lời bám sát tài liệu, tránh bịa (hallucination).

    Returns:
        Instance LLM, sẵn sàng gọi
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )
    return llm


def generate_answer(prompt: str) -> str:
    """
    Gửi prompt tới LLM và nhận câu trả lời.

    Args:
        prompt: nội dung prompt hoàn chỉnh (đã bao gồm context + câu hỏi)

    Returns:
        Câu trả lời dạng text thuần (đã xử lý các format trả về khác nhau)
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt rỗng, không thể gửi tới LLM")

    llm = get_llm()
    response = llm.invoke(prompt)

    return _extract_text(response.content)


REWRITE_PROMPT_TEMPLATE = """Bạn là trợ lý giúp cải thiện khả năng tìm kiếm tài liệu.
Cho câu hỏi gốc dưới đây, hãy viết lại thành {num_variants} câu hỏi khác nhau,
diễn đạt lại theo nhiều cách khác nhau nhưng vẫn giữ nguyên ý nghĩa và mục đích tìm kiếm.
Mục tiêu là tăng khả năng tìm được đoạn văn bản liên quan trong tài liệu, dù tài liệu
có thể dùng từ ngữ khác với câu hỏi gốc.

Câu hỏi gốc: {question}

Chỉ trả về đúng {num_variants} câu hỏi, mỗi câu 1 dòng, không đánh số, không giải thích thêm:"""


def rewrite_query(question: str, num_variants: int = 3) -> list[str]:
    """
    Sinh ra các biến thể diễn đạt khác nhau của 1 câu hỏi, dùng để mở rộng
    phạm vi tìm kiếm ngữ nghĩa (query expansion / rewriting).

    Args:
        question: câu hỏi gốc của người dùng
        num_variants: số lượng biến thể muốn sinh ra

    Returns:
        Danh sách câu hỏi, LUÔN bao gồm câu hỏi gốc ở vị trí đầu tiên,
        theo sau là các biến thể (nếu sinh thành công). Nếu LLM lỗi,
        trả về chỉ câu hỏi gốc (không làm gián đoạn luồng chính).
    """
    if not question or not question.strip():
        return [question]

    prompt = REWRITE_PROMPT_TEMPLATE.format(question=question, num_variants=num_variants)

    try:
        raw_output = generate_answer(prompt)
    except Exception as e:
        # Nếu rewriting lỗi vì bất kỳ lý do gì, không làm gián đoạn pipeline
        # chính — chỉ dùng câu hỏi gốc để search như bình thường.
        # In lỗi ra console để dễ debug (không hiện lên UI, tránh làm phiền user).
        logger.warning(f"rewrite_query thất bại, dùng câu hỏi gốc. Lỗi: {e}")
        print(f"[rewrite_query] LỖI: {e}")
        return [question]

    variants = [line.strip() for line in raw_output.split("\n") if line.strip()]
    # Loại bỏ số thứ tự nếu LLM lỡ thêm vào (VD "1. ", "- ")
    variants = [re.sub(r"^[\d\-\.\)\s]+", "", v).strip() for v in variants]
    variants = [v for v in variants if v]

    # Loại các dòng "giải thích/preamble" mà LLM đôi khi chèn thêm dù đã
    # được dặn không giải thích (VD "Dưới đây là 3 câu hỏi:", "Here are..."),
    # nhận diện bằng việc dòng đó không kết thúc như 1 câu hỏi thật và có
    # xu hướng dài bất thường kèm dấu ":". Đây chỉ là lọc tương đối, không
    # cần chính xác tuyệt đối — mục tiêu là tránh biến thể "rác" lọt vào
    # danh sách search.
    variants = [
        v for v in variants
        if not (v.endswith(":") or (len(v) > 15 and ":" in v.split()[0:3] and v.lower().startswith(("dưới đây", "here", "sau đây", "các câu"))))
    ]

    if len(variants) < num_variants:
        logger.warning(
            f"rewrite_query: chỉ parse được {len(variants)}/{num_variants} biến thể "
            f"mong đợi. Raw output: {raw_output!r}"
        )
        print(
            f"[rewrite_query] CẢNH BÁO: chỉ nhận được {len(variants)}/{num_variants} "
            f"biến thể hợp lệ sau khi parse."
        )

    all_queries = [question] + variants
    # Loại trùng lặp, giữ thứ tự
    seen = set()
    unique_queries = []
    for q in all_queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique_queries.append(q)

    return unique_queries


def _extract_text(content) -> str:
    """
    Chuẩn hóa content trả về từ LLM thành string thuần.
    SDK mới có thể trả về string, hoặc list các dict (kèm metadata khác
    như 'thought signature'), nên cần lọc lấy đúng phần text.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)

    return str(content)