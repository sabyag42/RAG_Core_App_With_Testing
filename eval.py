from rag import build_rag_chain, create_vector_store, chunk_documents, load_pdf, ask_question
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from langsmith import traceable
import pandas as pd

golden_dataset = [
    {
        "question": "What is the roaming charge for calls in EU/EEA countries?",
        "ground_truth": "€0.19 per minute"
    },
    {
        "question": "How do I cancel my plan?",
        "ground_truth": "My Odido App → Account Settings → Manage Plan → Cancel Plan Phone → Call 1234 Online → odido.nl/cancel In Store → with valid photo ID 30-day notice period applies"
    },
    {
        "question": "What happens if I pay my bill late?",
        "ground_truth": "€2.50 fee if not received within 7 days of due date, Service suspended after 30 days, €5.00 reconnection fee to restore service"
    },
    {
        "question": "What is the price of the Premium plan?",
        "ground_truth": "Premium is €44.99 with Unlimited data. No overage."
    },
    {
        "question": "What is the customer care phone number?",
        "ground_truth": "You can reach our customer care team at 1234 (free) or +31 20 123 4567 from Mon-Fri 8am-8pm, Sat 9am-5pm for assistance."
    }
]

@traceable(name="RAG Evaluation Pipeline")
def run_evaluation():

    # Step 1: Build RAG chain
    chain, retriever = build_rag_chain(
        create_vector_store(
            chunk_documents(
                load_pdf("data/odido_billing_faq.pdf")
            )
        )
    )

    # Step 2: Run each question through RAG
    questions, ground_truths, answers, contexts = [], [], [], []

    for item in golden_dataset:
        q = item['question']
        gt = item['ground_truth']

        ans, _ = ask_question(chain, retriever, q)
        source_docs = retriever.invoke(q)
        retrieved_contexts = [doc.page_content for doc in source_docs]

        questions.append(q)
        ground_truths.append(gt)
        answers.append(ans)
        contexts.append(retrieved_contexts)

    # Step 3: Wrap into HuggingFace Dataset
    # RAGAS 0.1.x expects these exact column names
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Step 4: Run RAGAS evaluation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings
    )

    # Step 5: Print results as table
    df = result.to_pandas()
    print("\n=== RAGAS EVALUATION RESULTS ===")
    print(df[['question', 'faithfulness', 'answer_relevancy', 'context_recall']])
    print(f"\nAverage Scores:")
    print(f"Faithfulness:      {df['faithfulness'].mean():.2f}")
    print(f"Answer Relevancy:  {df['answer_relevancy'].mean():.2f}")
    print(f"Context Recall:    {df['context_recall'].mean():.2f}")

    return result

if __name__ == "__main__":
    run_evaluation()