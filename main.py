from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.rag import rag
from app.classification import classify

app = FastAPI()

class RAGRequest(BaseModel):
    prompt: str

class ClassificationRequest(BaseModel):
    text: str

@app.post("/rag", response_model=List[str])
def rag_endpoint(request: RAGRequest):
    try:
        result = rag(request.prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classification", response_model=str)
def classification_endpoint(request: ClassificationRequest):
    try:
        result = classify(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
