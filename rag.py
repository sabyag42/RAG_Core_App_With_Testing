import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

load_dotenv()

def load_pdf(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from {file_path}")
    return pages

def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Vector store created with {vector_store.index.ntotal} vectors")
    return vector_store

def build_rag_chain(vector_store):
    template = """You are a helpful assistant.
Answer ONLY from the context below.
If answer not in context, say "I don't have that information."

Context: {context}
Question: {question}
Answer:"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain built successfully")
    return rag_chain, retriever

def ask_question(chain, retriever, question):
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    page_numbers = list(set([
        doc.metadata.get('page', 0) + 1
        for doc in source_docs
    ]))

    print(f"\nAnswer: {answer}")
    print(f"Page Numbers: {page_numbers}")
    return answer, page_numbers


if __name__ == "__main__":
    chain, retriever = build_rag_chain(
        create_vector_store(
            chunk_documents(
                load_pdf("data/odido_billing_faq.pdf")
            )
        )
    )

    questions = [
        "What is the roaming charge for EU countries?",
        "How do I cancel my plan?",
        "What happens if I pay my bill late?",
        "What is the price of the Premium plan?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        ask_question(chain, retriever, q)