from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.serving.inference import score_record

app = FastAPI(title="FinGuard Inference API")

REQS = Counter("requests_total", "Total requests")
LAT = Histogram("request_latency_seconds", "Latency")
@app.get("/")
def root():
    return {"message": "FinGuard Inference API is running 🚀"}

@app.get("/favicon.ico")
def favicon():
    # Avoid noisy 404s in logs
    return {"message": "No favicon"}

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type="text/plain")

@app.post("/score", response_model=PredictionResponse)
def score(txn: PredictionRequest):
    REQS.inc()
    with LAT.time():
        result = score_record(txn.dict(), explain=True)
    return result
