import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(

"pritamdeka/S-PubMedBert-MS-MARCO"

)

section_vectors = pickle.load(

open(

"embeddings/section_vectors.pkl",

"rb"

)

)

def retrieve_sections(

query,

candidate_chapters,

top_k=50

):

    q = model.encode(

        query,

        normalize_embeddings=True

    )

    scores=[]

    for key,vec in section_vectors.items():

        chapter_key=(

            key[0],

            key[1]

        )

        if chapter_key in candidate_chapters:

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