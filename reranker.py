import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# Load models and pre-computed vectors
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading models and vectors for fast reranking...")
model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device="cpu")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

# Load raw chunks to build lookup
CHUNKS_PATH = os.path.join(BASE_DIR, "processed", "chunks.pkl")
VECTORS_PATH = os.path.join(BASE_DIR, "embeddings", "chunk_vectors.pkl")

chunk_vectors_tensor = None
chunk_lookup = {}

if os.path.exists(CHUNKS_PATH) and os.path.exists(VECTORS_PATH):
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    with open(VECTORS_PATH, "rb") as f:
        chunk_vectors = pickle.load(f)
    chunk_vectors = np.array(chunk_vectors)
    norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
    chunk_vectors = chunk_vectors / np.where(norms == 0, 1.0, norms)
    chunk_vectors_tensor = torch.tensor(chunk_vectors).to("cpu")
    chunk_lookup = {chunk["text"].strip(): idx for idx, chunk in enumerate(chunks)}
    print("Pre-computed chunk vectors loaded for fast reranking.")
else:
    print("Warning: Pre-computed vectors not found, fallback to on-the-fly encoding.")

def rerank(query, docs, top_k=10):
    if len(docs) == 0:
        return []
        
    # If we have too many docs, filter using fast S-PubMedBert cosine similarity
    if len(docs) > 15 and chunk_vectors_tensor is not None:
        query_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        
        doc_embs_list = []
        for d in docs:
            text = d.get("text", "").strip()
            idx = chunk_lookup.get(text)
            if idx is not None:
                doc_embs_list.append(chunk_vectors_tensor[idx])
            else:
                doc_embs_list.append(model.encode(text, convert_to_tensor=True, normalize_embeddings=True))
        doc_embs = torch.stack(doc_embs_list)
        
        sims = util.cos_sim(query_emb, doc_embs).cpu().numpy().flatten()
        for doc, sim in zip(docs, sims):
            doc["_temp_sim"] = float(sim)
        
        docs.sort(key=lambda x: x["_temp_sim"], reverse=True)
        # Keep top 15 candidates for CrossEncoder reranking
        docs_to_rerank = docs[:15]
        # Clean up temp sim
        for d in docs:
            d.pop("_temp_sim", None)
    else:
        docs_to_rerank = docs

    pairs = [(query, doc["text"]) for doc in docs_to_rerank]
    scores = reranker.predict(pairs)
    
    for doc, score in zip(docs_to_rerank, scores):
        doc["score"] = float(score)
        
    # For any documents that were filtered out, assign a low score
    for doc in docs:
        if "score" not in doc:
            doc["score"] = -10.0
            
    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs[:top_k]