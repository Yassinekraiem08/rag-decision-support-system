from fastapi import FastAPI

app = FastAPI(title="AI Decision Support System")


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}