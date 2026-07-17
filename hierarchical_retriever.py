from retrieval.book_router import retrieve_books

from retrieval.chapter_router import retrieve_chapters

from retrieval.section_router import retrieve_sections

from retrieval.chunk_retriever import retrieve_candidate_chunks

from retrieval.reranker import rerank


def hierarchical_retrieval(

query

):

    books = retrieve_books(

        query,
        top_k=3

    )

    selected_books=[

        b[0]

        for b in books

    ]

    chapters = retrieve_chapters(

        query,

        selected_books,
        top_k=10

    )

    selected_chapters=[

        c[0]

        for c in chapters

    ]

    sections = retrieve_sections(

        query,

        selected_chapters,
        top_k=30

    )

    selected_sections=[

        s[0]

        for s in sections

    ]

    candidate_docs = retrieve_candidate_chunks(

        selected_sections

    )

    final_docs = rerank(

        query,

        candidate_docs,

        top_k=10

    )

    return {

        "books":

        books,

        "chapters":

        chapters,

        "sections":

        sections,

        "documents":

        final_docs

    }