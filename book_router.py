import os
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

BOOK_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "book_vectors.pkl"
)

print(BOOK_PATH)

model = SentenceTransformer(
    "pritamdeka/S-PubMedBert-MS-MARCO"
)

with open(
        BOOK_PATH,
        "rb"
) as f:

    book_vectors = pickle.load(
        f
    )

def retrieve_books(

        query,

        top_k=5

):

    q = model.encode(

        query,

        normalize_embeddings=True

    )

    scores=[]

    for book,vec in book_vectors.items():

        sim=np.dot(

            q,

            vec

        )

        scores.append(

            (

                book,

                float(sim)

            )

        )

    scores.sort(

        key=lambda x:x[1],

        reverse=True

    )

    return scores[:top_k]