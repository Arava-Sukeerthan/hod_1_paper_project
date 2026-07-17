import pickle

from sentence_transformers import SentenceTransformer

chunks=pickle.load(

open(

"processed/hierarchy.pkl",

"rb"

)

)

model=SentenceTransformer(

"pritamdeka/S-PubMedBert-MS-MARCO"

)

chapters={}

for c in chunks:

    key=(

        c["book"],

        c["chapter"]

    )

    chapters.setdefault(

        key,

        []

    ).append(

        c["text"]

    )

chapter_vectors={}

for key,texts in chapters.items():

    merged=" ".join(

        texts[:50]

    )

    chapter_vectors[key]=model.encode(

        merged,

        normalize_embeddings=True

    )

pickle.dump(

chapter_vectors,

open(

"chapter_vectors.pkl",

"wb"

)

)

print()

print(

len(

chapter_vectors

)

)