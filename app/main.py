from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="AI Decision Support System")

app.include_router(router)


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}