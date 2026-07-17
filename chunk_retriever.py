import pickle

chunks = pickle.load(

open(

"processed/chunks.pkl",

"rb"

)

)

def retrieve_candidate_chunks(

candidate_sections

):

    docs=[]

    candidate_sections=set(

        candidate_sections

    )

    for chunk in chunks:

        key=(

            chunk["book"],

            chunk["chapter"],

            chunk["section"]

        )

        if key in candidate_sections:

            docs.append(

                chunk

            )

    return docs
    
        

    