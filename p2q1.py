"""Parte 2 - Questão 1: Chatbot simples com LangChain + Fake LLM + FastAPI.

Executar: python p2q1.py   (porta 8003)

Por que Fake LLM? Permite testar rapidamente o fluxo da aplicação (rotas,
validação, prompts, chains) sem custo, sem internet, sem chave de API e com
respostas determinísticas, o que facilita testes automatizados.
"""
import unicodedata

import uvicorn
from fastapi import FastAPI
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Base de perguntas pré-definidas (chave normalizada -> resposta simulada).
FAQ = {
    "ola": "Olá! Sou um chatbot de demonstração. Como posso ajudar?",
    "qual e o seu nome": "Meu nome é FakeBot, um chatbot simulado com LangChain.",
    "o que e langchain": "LangChain é um framework para construir aplicações com LLMs.",
    "o que e fastapi": "FastAPI é um framework web moderno e rápido para criar APIs em Python.",
    "tchau": "Até logo! Foi um prazer conversar.",
}
DEFAULT_ANSWER = "Desculpe, não sei responder a essa pergunta (resposta simulada)."

prompt = PromptTemplate.from_template(
    "Você é um assistente prestativo. Responda à pergunta: {question}"
)

app = FastAPI(title="Chatbot com Fake LLM")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["O que é LangChain?"])


class ChatResponse(BaseModel):
    question: str
    answer: str


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().strip())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "".join(c for c in text if c.isalnum() or c.isspace()).strip()


def build_chain(question: str):
    """Monta a chain prompt | fake_llm | parser com a resposta escolhida."""
    answer = FAQ.get(normalize(question), DEFAULT_ANSWER)
    fake_llm = FakeListLLM(responses=[answer])
    return prompt | fake_llm | StrOutputParser()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    answer = await build_chain(req.message).ainvoke({"question": req.message})
    return ChatResponse(question=req.message, answer=answer)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
