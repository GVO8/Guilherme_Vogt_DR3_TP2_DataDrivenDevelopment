"""Parte 1 - Questão 1: Geração de texto com GPT-2 (HuggingFace) + FastAPI.

Executar: python p1q1.py   (porta 8001)
Docs interativas: http://localhost:8001/docs
"""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_NAME = "gpt2"
resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # O modelo é carregado uma única vez, na inicialização.
    resources["generator"] = pipeline("text-generation", model=MODEL_NAME)
    yield
    resources.clear()


app = FastAPI(title="GPT-2 Text Generation", lifespan=lifespan)


class GenerationRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Once upon a time"])
    max_new_tokens: int = Field(50, ge=1, le=200)


class GenerationResponse(BaseModel):
    input: str
    generated_text: str


@app.post("/generate", response_model=GenerationResponse)
def generate(req: GenerationRequest):
    try:
        out = resources["generator"](
            req.text,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=50256,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro na geração: {exc}")
    return GenerationResponse(input=req.text, generated_text=out[0]["generated_text"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
