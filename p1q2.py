"""Parte 1 - Questão 2: Tradução EN -> FR com Helsinki-NLP/opus-mt-en-fr + FastAPI.

Executar: python p1q2.py   (porta 8002)
Textos longos são divididos em sentenças e traduzidos em lote.
"""
import re
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources["translator"] = pipeline("translation_en_to_fr", model=MODEL_NAME)
    yield
    resources.clear()


app = FastAPI(title="Tradução EN -> FR (Helsinki-NLP)", lifespan=lifespan)


class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Hello, how are you?"])


class TranslationResponse(BaseModel):
    original: str
    translation: str


def split_sentences(text: str) -> list[str]:
    """Divide o texto em sentenças para respeitar o limite de tokens do modelo."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


@app.post("/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):
    sentences = split_sentences(req.text)
    if not sentences:
        raise HTTPException(status_code=422, detail="Texto vazio.")
    try:
        results = resources["translator"](sentences, max_length=512, batch_size=8)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro na tradução: {exc}")
    translation = " ".join(r["translation_text"] for r in results)
    return TranslationResponse(original=req.text, translation=translation)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
