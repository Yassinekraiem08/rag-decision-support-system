from app.utils.loaders import load_text_file

text = load_text_file("sample_doc.txt")

print("Loaded text:\n")
print(text)