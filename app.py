from fastapi import FastAPI
from pydantic import BaseModel
from rag import load_pdf, chunk_documents, create_vector_store, build_rag_chain, ask_question

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    page_numbers: list[int]

chain,retriever = build_rag_chain(
    create_vector_store(
        chunk_documents(
            load_pdf("data/odido_billing_faq.pdf")
        )
    )
)

@app.post("/ask", response_model=AnswerResponse)
def ask_endpoint(request: QuestionRequest):
    answer, page_numbers = ask_question(chain, retriever, request.question)
    return AnswerResponse(answer=answer, page_numbers=page_numbers)

"""
"FastAPI uses decorators to register Python functions as HTTP endpoints. The decorator specifies the HTTP method (POST), the route (/ask), and the response model which validates output and auto-generates API documentation. response_model also acts as a security layer — it strips any internal fields not in the model so they never leak to the caller."
"""