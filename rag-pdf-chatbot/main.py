"""
Entry point CLI — chạy RAG chatbot qua terminal, không cần giao diện web.
Dùng khi muốn demo nhanh hoặc debug mà không mở Streamlit.
"""

from app.chains.qa_chain import ask


def main():
    print("=" * 50)
    print("RAG PDF Chatbot — chế độ dòng lệnh")
    print("Gõ 'exit' hoặc 'quit' để thoát")
    print("=" * 50)

    while True:
        question = input("\nCâu hỏi của bạn: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Tạm biệt!")
            break

        if not question:
            continue

        try:
            result = ask(question)
            print(f"\nTrả lời: {result['answer']}")
        except Exception as e:
            print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()