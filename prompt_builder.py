from typing import List, Dict, Any

def build_prompt(query: str, retrieved_documents: List[Dict[str, Any]]) -> str:
    """
    Constructs a structured, publication-level medical RAG prompt template.
    
    Injects clear guidelines to prevent hallucination, forcing the model to adhere 
    strictly to the provided literature snippets and to output a standardized fallback
    if the documentation is insufficient.
    
    Args:
        query: The user's clinical query.
        retrieved_documents: A list of retrieved literature passages/evidence chunks.
        
    Returns:
        A formatted prompt string to feed to the language model.
    """
    # 1. Format the evidence passages with clear metadata headers
    evidence_text = ""
    for i, doc in enumerate(retrieved_documents):
        book = doc.get("book", "Unknown Source")
        chapter = doc.get("chapter", "N/A")
        section = doc.get("section", "N/A")
        text = doc.get("text", "").strip()
        
        evidence_text += f"Evidence Snippet [{i+1}] (Source: {book} | Chapter: {chapter} | Section: {section}):\n"
        evidence_text += f"{text}\n\n"
        
    # 2. Assemble the scientific clinical assistant instructions
    prompt = f"""You are an elite clinical assistant and expert medical search engine. Your task is to provide publication-grade, evidence-grounded answers based strictly on the retrieved medical literature provided below.

=== STRICT CLINICAL DIRECTIVES ===
1. ANSWER GROUNDING: Answer the query using ONLY the explicitly stated facts in the Medical Evidence section below. Do not use outside medical knowledge, general training data, speculative hypotheses, or extrapolation.
2. EVIDENCE INSUFFICIENCY RULE: If the retrieved Medical Evidence is insufficient, incomplete, or does not contain direct answers to the query, you must output exactly this string and nothing else:
"Insufficient evidence available in retrieved documents."
3. NO HALLUCINATION: Every clinical statement, diagnostic criteria, or anatomical detail must be fully supported by the evidence. Do not guess or formulate answers that are not directly supported.
4. STRUCTURE: You must partition your response exactly into the three headers below:

[ANSWER]
<Provide a concise, direct, and factual answer to the clinical question, fully grounded in the snippets.>

[CLINICAL EXPLANATION]
<Explain the underlying clinical rationale, pathophysiology, or diagnostic details supporting the answer, relying solely on the provided snippets.>

[KEY POINTS]
- <Bullet point 1 detailing a key medical takeaway.>
- <Bullet point 2 detailing a key medical takeaway.>

=== MEDICAL EVIDENCE ===
{evidence_text.strip()}

=== CLINICAL QUESTION ===
{query}

Provide your structured clinical response below:
"""
    return prompt