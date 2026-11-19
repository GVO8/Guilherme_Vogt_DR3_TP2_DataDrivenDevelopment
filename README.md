# TP2 - FastAPI e LangChain com LLMs

APIs simples de geração de texto, chatbot e tradução usando FastAPI, LangChain e modelos HuggingFace/Groq.

| Arquivo | Descrição | Porta |
|---|---|---|
| `p1q1.py` | GPT-2 (HuggingFace) com FastAPI | 8001 |
| `p1q2.py` | Tradução EN→FR (Helsinki-NLP) com FastAPI | 8002 |
| `p2q1.py` | Chatbot com Fake LLM (LangChain) | 8003 |
| `p2q2.py` | Tradução EN→FR com LangChain + Groq | 8004 |
| `p2q3.py` | Tradução EN→DE com LangChain + HuggingFace | 8005 |

## Instalação
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # preencha GROQ_API_KEY (somente p2q2.py)
```

## Execução e exemplos
```bash
python p1q1.py
curl -X POST localhost:8001/generate -H "Content-Type: application/json" -d '{"text":"Once upon a time"}'

python p1q2.py
curl -X POST localhost:8002/translate -H "Content-Type: application/json" -d '{"text":"Hello, how are you?"}'

python p2q1.py
curl -X POST localhost:8003/chat -H "Content-Type: application/json" -d '{"message":"O que é LangChain?"}'

python p2q2.py
curl -X POST localhost:8004/translate -H "Content-Type: application/json" -d '{"text":"Good morning!"}'

python p2q3.py
curl -X POST localhost:8005/translate -H "Content-Type: application/json" -d '{"text":"The weather is nice today."}'
```
Documentação interativa (Swagger) em `http://localhost:<porta>/docs`.

Na primeira execução, os modelos HuggingFace são baixados automaticamente.
