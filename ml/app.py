import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

FRONTENDS = os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else ["*"]

app = FastAPI(title="MyKereta ML API")

# Useful for local testing; in prod we proxy via Node so CORS isn't hit.
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTENDS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

class TextIn(BaseModel):
    text: str

@app.post("/ml/summarize")
def summarize(payload: TextIn):
    # TODO: call your model here
    text = payload.text
    return {"summary": (text[:120] + "...") if len(text) > 120 else text}

@app.post("/ml/plate")
async def plate(file: UploadFile = File(...)):
    # TODO: run OCR/plate detection on file
    # file.file is a SpooledTemporaryFile
    return {"plate": "WXX 1234"}  # placeholder