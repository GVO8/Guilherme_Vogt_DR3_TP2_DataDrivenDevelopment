"""Parte 2 - Questão 2: Tradução EN -> FR com LangChain + Groq (API compatível com OpenAI).

Executar: GROQ_API_KEY=... python p2q2.py   (porta 8004)

Para usar a OpenAI diretamente, troque ChatGroq por
`from langchain_openai import ChatOpenAI` e use OPENAI_API_KEY.
"""
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Defina a variável de ambiente GROQ_API_KEY (veja .env.example).")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um tradutor profissional. Traduza o texto do inglês para o francês. "
            "Responda APENAS com a tradução, sem comentários.",
        ),
        ("human", "{text}"),
    ]
)
llm = ChatGroq(model=MODEL_NAME, temperature=0, max_retries=2, timeout=30)
chain = prompt | llm | StrOutputParser()

app = FastAPI(title="Tradução EN -> FR (LangChain + Groq)")


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Good morning, how are you?"])


class TranslationResponse(BaseModel):
    original: str
    translation: str


@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    try:
        translation = await chain.ainvoke({"text": req.text})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Erro ao chamar o LLM: {exc}")
    return TranslationResponse(original=req.text, translation=translation.strip())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
