"""
Đánh giá chất lượng hệ thống RAG bằng RAGAS.

Khác với bộ test case thủ công trong README (đọc bằng mắt), script này
chạy pipeline thật (chat_service.answer_question) trên 1 bộ câu hỏi có
sẵn đáp án đúng (ground truth), rồi dùng RAGAS để tính các metric khách
quan:

- faithfulness       : câu trả lời có bám sát context lấy được không,
                        hay tự bịa thêm (đo hallucination trực tiếp)
- answer_relevancy    : câu trả lời có thực sự trả lời đúng câu hỏi không
- context_precision   : trong các đoạn context lấy được, bao nhiêu % thực
                         sự liên quan (đo chất lượng retrieval)
- context_recall      : hệ thống có lấy được ĐỦ context cần thiết để trả
                         lời đúng không (so với ground truth)

Cách dùng:
    1. Sửa EVAL_DATASET bên dưới (hoặc load từ file JSON riêng — xem
       hàm load_dataset_from_json).
    2. Đảm bảo đã ingest sẵn tài liệu cần test vào vector store.
    3. Chạy: python evaluate_ragas.py
    4. Kết quả in ra console + lưu CSV chi tiết từng câu hỏi vào
       eval_results.csv để soi lại câu nào yếu.

Lưu ý:
- Script này gọi API thật (Gemini) cho từng câu hỏi + RAGAS cũng gọi LLM
  để chấm điểm answer_relevancy/context_precision → TỐN QUOTA, không
  chạy trong CI/pytest thường xuyên. Coi đây là benchmark định kỳ, khác
  với unit test (tests/) chạy nhanh không tốn quota.
- RAGAS mặc định dùng OpenAI để làm "giám khảo" (LLM chấm điểm). Vì
  project này dùng Gemini, cần cấu hình lại evaluator LLM/embeddings
  sang Gemini để không phải xin thêm OPENAI_API_KEY (xem phần cấu hình
  bên dưới).
"""

import json
import logging
import sys
from pathlib import Path

# Cho phép chạy script này từ thư mục gốc project (import được app.*)
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 1. BỘ CÂU HỎI ĐÁNH GIÁ (ground truth tự viết dựa trên tài liệu test)
# ============================================================
# Mỗi entry cần: question + ground_truth (đáp án đúng, tự viết tay dựa
# trên nội dung tài liệu thật đã ingest). Có thể thêm/sửa tùy tài liệu
# bạn đang dùng để test (VD báo cáo NCKH NOMA nhắc trong README).
#
# Khuyến nghị: nên có cả câu hỏi "bẫy" (không có trong tài liệu) để xem
# faithfulness có giữ vững không khi hệ thống đúng là từ chối trả lời.

EVAL_DATASET = [
    {
        "question": "Mã số đề tài là gì?",
        "ground_truth": "Điền đáp án đúng theo tài liệu thật bạn đang test.",
    },
    {
        "question": "Phương trình tính SINR tại U1, U2 là gì?",
        "ground_truth": "Điền công thức đúng theo tài liệu thật.",
    },
    {
        "question": "Kinh phí thực hiện đề tài là bao nhiêu?",
        "ground_truth": "Không có thông tin về kinh phí trong tài liệu.",
    },
    # ... thêm câu hỏi khác tương ứng tài liệu bạn dùng để eval
]


def load_dataset_from_json(path: str | Path) -> list[dict]:
    """
    Tùy chọn: load bộ câu hỏi từ file JSON thay vì hardcode trong file
    này, để dễ mở rộng/tái sử dụng cho nhiều tài liệu khác nhau.

    Format file JSON mong đợi:
    [
        {"question": "...", "ground_truth": "..."},
        ...
    ]
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. CHẠY PIPELINE THẬT ĐỂ THU THẬP DỮ LIỆU CHO RAGAS
# ============================================================

def run_pipeline_on_dataset(dataset: list[dict], top_k: int = 4) -> dict:
    """
    Chạy answer_question() thật cho từng câu hỏi trong dataset, gom lại
    thành format mà RAGAS cần: question, answer, contexts, ground_truth.

    Args:
        dataset: danh sách {"question", "ground_truth"}
        top_k: số đoạn context lấy ra mỗi câu hỏi (nên khớp với config
               thật đang dùng trong app để kết quả eval phản ánh đúng
               hệ thống thực tế)

    Returns:
        dict các list song song, đúng format ragas.EvaluationDataset /
        datasets.Dataset:
            {
                "question": [...],
                "answer": [...],
                "contexts": [[...], [...], ...],
                "ground_truth": [...],
            }
    """
    from app.services.chat_service import answer_question

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        logger.info(f"[{i}/{len(dataset)}] Đang chạy: {question!r}")

        result = answer_question(question, top_k=top_k)

        if result.get("error"):
            logger.warning(f"  -> Lỗi khi trả lời, bỏ qua câu này: {question!r}")
            continue

        questions.append(question)
        answers.append(result["answer"])
        # answer_question() trả sources đã bị truncate (300 ký tự) để
        # hiển thị UI gọn — với RAGAS nên dùng context ĐẦY ĐỦ, không cắt,
        # vì cắt bớt context có thể làm faithfulness/context_precision
        # bị đánh giá sai (thiếu thông tin thật sự đã đưa vào prompt).
        # Nên gọi thẳng qa_chain.ask() ở đây thay vì chat_service để lấy
        # context chưa truncate.
        contexts_list.append(result["sources"])
        ground_truths.append(item["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


def run_pipeline_on_dataset_full_context(dataset: list[dict], top_k: int = 4) -> dict:
    """
    Bản thay thế cho run_pipeline_on_dataset(), dùng thẳng qa_chain.ask()
    thay vì chat_service.answer_question() để lấy context ĐẦY ĐỦ (không
    bị truncate 300 ký tự như chat_service làm để hiển thị UI).

    Nên dùng hàm này cho việc eval thay vì hàm ở trên, vì RAGAS cần biết
    chính xác context nào đã thực sự được đưa vào prompt.
    """
    from app.chains.qa_chain import ask

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        logger.info(f"[{i}/{len(dataset)}] Đang chạy: {question!r}")

        try:
            result = ask(question, top_k=top_k)
        except Exception as e:
            logger.warning(f"  -> Lỗi khi trả lời, bỏ qua câu này: {question!r} ({e})")
            continue

        questions.append(question)
        answers.append(result["answer"])
        contexts_list.append(result["sources"])
        ground_truths.append(item["ground_truth"])

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


# ============================================================
# 3. CHẠY RAGAS
# ============================================================

def evaluate_with_ragas(data: dict):
    """
    Chạy RAGAS trên dữ liệu đã thu thập, dùng Gemini làm evaluator LLM
    thay vì mặc định OpenAI (project này không dùng OpenAI key).

    Returns:
        ragas EvaluationResult (có thể .to_pandas() để lưu CSV)
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    from app.config import GOOGLE_API_KEY, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME

    # RAGAS cần 1 LLM + 1 embedding model để tự chấm điểm (VD: đo answer
    # có "relevant" với question không). Dùng lại đúng Gemini model đang
    # cấu hình trong app.config để nhất quán, không cần thêm API key khác.
    ragas_llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,  # chấm điểm cần ổn định, không cần sáng tạo
    )
    ragas_embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
    )

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return result


# ============================================================
# 4. MAIN
# ============================================================

def main():
    dataset = EVAL_DATASET
    # Nếu muốn load từ file JSON thay vì hardcode ở trên, bỏ comment:
    # dataset = load_dataset_from_json("eval_dataset.json")

    if not dataset:
        logger.error("EVAL_DATASET rỗng — hãy điền câu hỏi + ground_truth trước khi chạy.")
        return

    logger.info(f"Chạy pipeline thật trên {len(dataset)} câu hỏi...")
    data = run_pipeline_on_dataset_full_context(dataset)

    if not data["question"]:
        logger.error("Không có câu hỏi nào chạy thành công, dừng lại.")
        return

    logger.info("Chạy RAGAS để chấm điểm...")
    result = evaluate_with_ragas(data)

    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ RAGAS")
    print("=" * 50)
    print(result)

    # Lưu chi tiết từng câu hỏi ra CSV để soi lại câu nào điểm thấp
    df = result.to_pandas()
    out_path = Path(__file__).resolve().parent / "eval_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu chi tiết từng câu hỏi vào: {out_path}")


if __name__ == "__main__":
    main()