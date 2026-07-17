from typing import List, Dict, Any

def add_citations(generated_answer: str, retrieved_documents: List[Dict[str, Any]]) -> str:
    """
    Appends a formatted, deduplicated publication-grade References section
    to the generated clinical answer.
    
    Args:
        generated_answer: The grounded medical answer string from the model.
        retrieved_documents: List of retrieved literature documents containing book and chapter info.
        
    Returns:
        The text answer appended with the structured References block.
    """
    # If the answer indicates insufficient evidence or is empty, we do not append references
    cleaned_answer = generated_answer.strip()
    if not cleaned_answer or "Insufficient evidence available in retrieved documents." in cleaned_answer:
        return cleaned_answer

    # 1. Deduplicate references by tracking (book, chapter) tuples
    seen_references = set()
    unique_references = []
    
    for doc in retrieved_documents:
        book = doc.get("book")
        chapter = doc.get("chapter")
        
        # Verify book metadata is present
        if not book:
            continue
            
        book_clean = str(book).strip()
        
        # Build reference identifier
        ref_key = (book_clean, chapter)
        if ref_key not in seen_references:
            seen_references.add(ref_key)
            unique_references.append(ref_key)
            
    # 2. If no valid references were identified, return the answer unchanged
    if not unique_references:
        return cleaned_answer
        
    # 3. Format References section in publication-quality design
    references_block = "\n\nReferences\n"
    for book_name, chapter_num in unique_references:
        references_block += f"\n{book_name}\n"
        if chapter_num is not None:
            references_block += f"Chapter {chapter_num}\n"
            
    return cleaned_answer + references_block