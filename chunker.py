import pickle

data=pickle.load(

open(

"../processed/hierarchy.pkl",

"rb"

)

)

chunks=[]

CHUNK_SIZE=300

for item in data:

    words=item["text"].split()

    for i in range(

            0,

            len(words),

            CHUNK_SIZE

    ):

        chunk=" ".join(

            words[

                i:i+CHUNK_SIZE

            ]

        )

        if len(chunk)>50:

            chunks.append(

                {

                    "book":

                    item["book"],

                    "chapter":

                    item["chapter"],

                    "section":

                    item["section"],

                    "text":

                    chunk

                }

            )

pickle.dump(

chunks,

open(

"../processed/chunks.pkl",

"wb"

)

)

print()

print(

len(chunks),

"chunks"

)

print(

"Saved chunks.pkl"

)