import pickle
import re
from pathlib import Path

###########################################################
# CONFIGURATION
###########################################################

WINDOW_SIZE = 1200
OVERLAP = 200

###########################################################
# LOAD DOCUMENTS
###########################################################

BASE_DIR = Path(__file__).resolve().parent.parent

docs = pickle.load(

    open(

        BASE_DIR / "processed" / "raw_documents.pkl",

        "rb"

    )

)

hierarchy = []

###########################################################
# BUILD HIERARCHY
###########################################################

for doc in docs:

    book = doc["book"]

    text = doc["text"]

    chapters = re.split(

        r'Chapter\s+\d+',

        text

    )

    print()

    print(book)

    print(

        len(chapters),

        "chapters"

    )

    for cid, chapter in enumerate(

            chapters

    ):

        chapter = chapter.strip()

        if len(chapter) < 100:

            continue

        sections = []

        start = 0

        while start < len(chapter):

            end = min(

                start + WINDOW_SIZE,

                len(chapter)

            )

            section = chapter[start:end]

            section = section.strip()

            if len(section) > 100:

                sections.append(

                    section

                )

            start += (

                WINDOW_SIZE -

                OVERLAP

            )

        ###################################################

        for sid, section in enumerate(

                sections

        ):

            hierarchy.append(

                {

                    "book":

                        book,

                    "chapter":

                        cid,

                    "section":

                        sid,

                    "text":

                        section

                }

            )

###########################################################
# SAVE
###########################################################

pickle.dump(

    hierarchy,

    open(

        BASE_DIR /

        "processed" /

        "hierarchy.pkl",

        "wb"

    )

)

print()

print(

    len(

        hierarchy

    ),

    "sections"

)

print(

    "Saved hierarchy.pkl"

)