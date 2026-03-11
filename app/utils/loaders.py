from fastapi import UploadFile
from pypdf import PdfReader
import os


def clean_text(text: str) -> str:
    return text.replace("\x00", "")


def load_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return clean_text(file.read())


def load_pdf_file(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return clean_text(text)


async def load_uploaded_text_file(file: UploadFile) -> str:
    content = await file.read()
    return clean_text(content.decode("utf-8"))


async def load_uploaded_pdf_file(file: UploadFile) -> str:
    content = await file.read()

    temp_path = "temp_upload.pdf"
    with open(temp_path, "wb") as temp_file:
        temp_file.write(content)

    reader = PdfReader(temp_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return clean_text(text)