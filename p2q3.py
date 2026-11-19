"""Parte 2 - Questão 3: Tradução EN -> DE com LangChain + HuggingFace (opus-mt-en-de) + FastAPI.

Executar: python p2q3.py   (porta 8005)
"""
import re
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    hf_pipe = pipeline("translation", model=MODEL_NAME, max_length=512)
    # LangChain gerencia a chamada ao modelo HuggingFace local.
    resources["llm"] = HuggingFacePipeline(pipeline=hf_pipe)
    yield
    resources.clear()


app = FastAPI(title="Tradução EN -> DE (LangChain + HuggingFace)", lifespan=lifespan)


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["The weather is nice today."])


class TranslationResponse(BaseModel):
    original: str
    translation: str


def split_sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]


@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    sentences = split_sentences(req.text)
    if not sentences:
        raise HTTPException(status_code=422, detail="Texto vazio.")
    try:
        # .invoke é síncrono/CPU-bound: roda em threadpool para não bloquear o event loop.
        outputs = await run_in_threadpool(resources["llm"].batch, sentences)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro na tradução: {exc}")
    return TranslationResponse(original=req.text, translation=" ".join(o.strip() for o in outputs))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
