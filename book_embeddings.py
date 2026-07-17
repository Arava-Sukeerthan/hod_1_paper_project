import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

chunks = pickle.load(open(os.path.join(BASE_DIR, "processed", "chunks.pkl"), "rb"))

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(

"pritamdeka/S-PubMedBert-MS-MARCO"

)

books={}

for c in chunks:

    book=c["book"]

    books.setdefault(

        book,

        []

    ).append(

        c["text"]

    )

book_vectors={}

for book,texts in books.items():

    merged=" ".join(

        texts[:200]

    )

    emb=model.encode(

        merged,

        normalize_embeddings=True

    )

    book_vectors[book]=emb

pickle.dump(

book_vectors,

open(

"book_vectors.pkl",

"wb"

)

)

print()

print(

len(

book_vectors

),

"books"

)

print()

print(

"Saved"

)