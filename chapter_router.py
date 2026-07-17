import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHAPTER_PATH = os.path.join(
    BASE_DIR,
    "embeddings",
    "chapter_vectors.pkl"
)

with open(
        CHAPTER_PATH,
        "rb"
) as f:

    chapter_vectors = pickle.load(
        f
    )

model=SentenceTransformer(

"pritamdeka/S-PubMedBert-MS-MARCO"

)

def retrieve_chapters(

query,

candidate_books,

top_k=20

):

    q=model.encode(

        query,

        normalize_embeddings=True

    )

    scores=[]

    for key,vec in chapter_vectors.items():

        book=key[0]

        if book in candidate_books:

            sim=np.dot(

                q,

                vec

            )

            scores.append(

                (

                    key,

                    float(sim)

                )

            )

    scores.sort(

        key=lambda x:x[1],

        reverse=True

    )

    return scores[:top_k]