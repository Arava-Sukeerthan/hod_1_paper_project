import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def save_for_evaluation(
    question: str,
    grounded_response: Any,
    output_dir: str = "outputs",
    filename: str = "generated_answers.json"
) -> Dict[str, Any]:
    """
    Saves RAG generation output records to a target JSON file formatted for evaluation.
    
    This function appends evaluation data elements (e.g. faithfulness, BLEU, hallucination checks)
    to the target dataset to support model benchmarking.
    
    Args:
        question: The input query string.
        grounded_response: The GroundedResponse object returned from grounded_generation.
        output_dir: Target directory path. Defaults to "outputs".
        filename: Target filename. Defaults to "generated_answers.json".
        
    Returns:
        A dictionary containing the parsed evaluation record.
    """
    try:
        # Create output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # 1. Resolve structured components from GroundedResponse
        is_sufficient = getattr(grounded_response, "is_sufficient", True)
        confidence = getattr(grounded_response, "confidence", 0.0)
        
        if not is_sufficient:
            answer = "Insufficient evidence available in retrieved documents."
            explanation = ""
            key_points = []
            references_formatted = []
        else:
            answer = getattr(grounded_response, "answer", "")
            explanation = getattr(grounded_response, "clinical_explanation", "")
            key_points = getattr(grounded_response, "key_points", [])
            
            # Format references as clean lists of strings
            references_raw = getattr(grounded_response, "references", [])
            references_formatted = []
            for ref in references_raw:
                book = ref.get("book", "Unknown Book")
                chapter = ref.get("chapter")
                ref_str = f"{book}"
                if chapter is not None:
                    ref_str += f" Chapter {chapter}"
                references_formatted.append(ref_str)
                
        # 2. Build the standard RAGAS/BERT evaluation record schema
        eval_record = {
            "question": question,
            "answer": answer,
            "clinical_explanation": explanation,
            "key_points": key_points,
            "references": references_formatted,
            "metadata": {
                "confidence": confidence,
                "is_sufficient": is_sufficient
            }
        }
        
        # 3. Read existing evaluation runs to support file appending
        existing_records: List[Dict[str, Any]] = []
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        existing_records = json.loads(content)
                        if not isinstance(existing_records, list):
                            existing_records = []
            except Exception as read_err:
                logger.warning(f"Could not read existing evaluation file: {read_err}. Overwriting.")
                existing_records = []
                
        # Append the new record to evaluation dataset
        existing_records.append(eval_record)
        
        # 4. Save formatted dataset
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing_records, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Evaluation record successfully saved to {filepath}")
        return eval_record
        
    except Exception as e:
        logger.error(f"Failed to save evaluation record: {e}", exc_info=True)
        return {}
