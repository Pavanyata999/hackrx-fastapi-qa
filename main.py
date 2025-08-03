from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import os, pdfplumber, requests, asyncio
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# ==============================
# LOAD ENVIRONMENT VARIABLES
# ==============================
load_dotenv()  # Loads from .env file

HACKRX_BEARER_TOKEN = os.getenv("HACKRX_BEARER_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

app = FastAPI(title="HackRx Ultra-Fast API")
security = HTTPBearer()

class HackRxRequest(BaseModel):
    questions: List[str]

# ==============================
# PROMPT FOR HACKRX STYLE ANSWERS
# ==============================
QA_PROMPT = PromptTemplate(
    template=(
        "You are an insurance policy assistant for HackRx.\n"
        "Use only the provided policy text to answer.\n"
        "Give the answer in one short, complete, and official-sounding sentence.\n"
        "Do not use bullet points, lists, or unnecessary explanations.\n"
        "Match the tone of the HackRx sample answers exactly.\n"
        "If the answer is not present, reply exactly: Not mentioned in the policy.\n\n"
        "Policy text:\n{context}\n\n"
        "Q: {question}\nA:"
    ),
    input_variables=["context", "question"]
)

# ==============================
# PRELOAD PDF ON STARTUP
# ==============================
PDF_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

print("Downloading and embedding PDF...")
pdf_path = "policy.pdf"
r = requests.get(PDF_URL, timeout=15)
with open(pdf_path, "wb") as f:
    f.write(r.content)

def extract_pdf_text(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                text += txt + "\n"
    return text.strip()

policy_text = extract_pdf_text(pdf_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
chunks = splitter.split_text(policy_text)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGroq(model="llama3-70b-8192", temperature=0, max_tokens=256)

print("Policy embedded & retriever ready!")

# ==============================
# API ENDPOINT
# ==============================
@app.post("/hackrx/run")
async def hackrx_run(
    body: HackRxRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials.strip()
    if token.startswith("Bearer "):
        token = token.split("Bearer ")[1].strip()
    if token != HACKRX_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    async def answer_question(q):
        docs = retriever.get_relevant_documents(q)
        context = "\n".join([d.page_content for d in docs])
        prompt = QA_PROMPT.format(context=context, question=q)
        response = await llm.ainvoke(prompt)
        return response.content.strip()

    answers = await asyncio.gather(*(answer_question(q) for q in body.questions))
    return {"answers": answers}
