import os
import pickle
from pypdf import PdfReader

# Configuration
BOOK_DIR = "../data/books"
SAVE_DIR = "../processed"
SAVE_PATH = os.path.join(SAVE_DIR, "raw_documents.pkl")

# Ensure output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

documents = []
files = os.listdir(BOOK_DIR)

print(f"\nFound PDFs: {files}\n")

for file in files:
    if file.endswith(".pdf"):
        path = os.path.join(BOOK_DIR, file)
        print(f"Reading: {file}")
        
        try:
            reader = PdfReader(path)
            text = ""
            total = len(reader.pages)
            print(f"Pages: {total}")
            
            for i, page in enumerate(reader.pages):
                try:
                    t = page.extract_text()
                    if t:
                        text += t
                except:
                    continue
                
                if i % 100 == 0:
                    print(f"Progress: {i} / {total}")
            
            documents.append({"book": file, "text": text})
            print("Finished processing file.")
            
        except Exception as e:
            print(f"\nError processing {file}: {e}")

# Save the results
with open(SAVE_PATH, "wb") as f:
    pickle.dump(documents, f)

print(f"\n{len(documents)} books loaded and saved to {SAVE_PATH}")