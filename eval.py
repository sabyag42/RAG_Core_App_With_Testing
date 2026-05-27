from rag import build_rag_chain, create_vector_store, chunk_documents, load_pdf, ask_question
from langchain_openai import ChatOpenAI
from langsmith import traceable
import pandas as pd

golden_dataset = [
    {"question": "What is the roaming charge for calls in EU/EEA countries?",
     "ground_truth": "€0.19 per minute"},
    {"question": "How do I cancel my plan?",
     "ground_truth": "30-day notice period. Via App, Phone 1234, Online or In Store."},
    {"question": "What happens if I pay my bill late?",
     "ground_truth": "€2.50 fee after 7 days. Suspended after 30 days. €5.00 reconnection fee."},
    {"question": "What is the price of the Premium plan?",
     "ground_truth": "€44.99 with Unlimited data. No overage."},
    {"question": "What is the customer care phone number?",
     "ground_truth": "1234 (free) or +31 20 123 4567. Mon-Fri 8am-8pm, Sat 9am-5pm."}
]

judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def score_faithfulness(answer, contexts):
    """Does the answer only use info from the retrieved chunks?"""
    context_text = "\n".join(contexts)
    prompt = f"""Rate if this answer is faithful to the context (only uses info from context).
Context: {context_text}
Answer: {answer}
Return ONLY a number between 0 and 1. 1=fully faithful, 0=not faithful."""
    result = judge_llm.invoke(prompt)
    try:
        return float(result.content.strip())
    except:
        return 0.0

def score_relevancy(question, answer):
    """Does the answer actually address the question?"""
    prompt = f"""Rate how relevant this answer is to the question.
Question: {question}
Answer: {answer}
Return ONLY a number between 0 and 1. 1=perfectly relevant, 0=completely irrelevant."""
    result = judge_llm.invoke(prompt)
    try:
        return float(result.content.strip())
    except:
        return 0.0

def score_correctness(answer, ground_truth):
    """Does the answer match the ground truth?"""
    prompt = f"""Rate how correct this answer is compared to the ground truth.
Ground Truth: {ground_truth}
Answer: {answer}
Return ONLY a number between 0 and 1. 1=perfectly correct, 0=completely wrong."""
    result = judge_llm.invoke(prompt)
    try:
        return float(result.content.strip())
    except:
        return 0.0

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

    # Step 2: Run each question and evaluate
    results = []

    for item in golden_dataset:
        q = item['question']
        gt = item['ground_truth']

        ans, _ = ask_question(chain, retriever, q)
        source_docs = retriever.invoke(q)
        contexts = [doc.page_content for doc in source_docs]

        # Score with LLM-as-judge
        faithfulness = score_faithfulness(ans, contexts)
        relevancy = score_relevancy(q, ans)
        correctness = score_correctness(ans, gt)

        results.append({
            "question": q,
            "answer": ans,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "correctness": correctness
        })

        print(f"✓ Evaluated: {q[:50]}...")

    # Step 3: Print results
    df = pd.DataFrame(results)
    print("\n=== EVALUATION RESULTS ===")
    print(df[['question', 'faithfulness', 'answer_relevancy', 'correctness']].to_string())
    print(f"\nAverage Scores:")
    print(f"Faithfulness:      {df['faithfulness'].mean():.2f}")
    print(f"Answer Relevancy:  {df['answer_relevancy'].mean():.2f}")
    print(f"Correctness:       {df['correctness'].mean():.2f}")


    # Step 4: Check thresholds  ← ADD IT HERE
    THRESHOLDS = {
        "faithfulness": 0.60,
        "answer_relevancy": 0.60,
        "correctness": 0.60
    }
    failed = False
    print("\n=== THRESHOLD CHECK ===")
    for metric, threshold in THRESHOLDS.items():
        score = df[metric].mean()
        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        print(f"{metric}: {score:.2f} (min: {threshold}) {status}")
        if score < threshold:
            failed = True

    if failed:
        print("\n❌ EVALUATION FAILED — scores below threshold")
        import sys
        sys.exit(1)
    else:
        print("\n✅ EVALUATION PASSED — safe to deploy")

    return df




if __name__ == "__main__":
    run_evaluation()


