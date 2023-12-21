# main.py
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from app.qa_processor import QAProcessor

app = FastAPI()

@app.post("/qa")
async def run_qa(pdf_file: UploadFile = File(...), questions: List[str] = None):
    if questions is None:
        questions = []

    processor = QAProcessor(pdf_file.filename, questions)

    # Save the uploaded PDF file to a temporary location
    with open(pdf_file.filename, "wb") as pdf:
        pdf.write(pdf_file.file.read())

    try:
        # Process PDF and questions
        return StreamingResponse(
            processor.process_qa(),
            media_type="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
