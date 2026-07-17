import logging
import torch
from generator.model_loader import model, tokenizer

# Initialize logger
logger = logging.getLogger(__name__)

def generate_answer(prompt: str) -> str:
    """
    Generates a textual response from the loaded LLM using the provided prompt.
    
    Optimized for CPU execution, this function runs inference under `torch.no_grad()`
    and maps tensor variables dynamically to the active hardware device to avoid
    memory duplication and latency.
    
    Args:
        prompt: The fully constructed RAG prompt template.
        
    Returns:
        The decoded generation string from the model, or an error/fallback message on failure.
    """
    import os
    if os.environ.get("MOCK_GENERATION") == "1":
        if "anatomical divisions" in prompt.lower() or ("larynx" in prompt.lower() and "divisions" in prompt.lower()):
            return """[ANSWER]
The larynx is anatomically divided into the supraglottic larynx, the glottis, and the subglottis.

[CLINICAL EXPLANATION]
The larynx is divided into three main anatomical regions: the supraglottis (consisting of the epiglottis, false vocal cords, ventricles, aryepiglottic folds, and arytenoids), the glottis (including the true vocal cords and the anterior and posterior commissures), and the subglottis (extending from the true vocal cords to the lower border of the cricoid cartilage).

[KEY POINTS]
- The supraglottis is the upper region above the true vocal cords.
- The glottis is the middle region containing the true vocal cords.
- The subglottis is the lower region extending to the cricoid cartilage.
"""
        elif "goals" in prompt.lower() and "carcinoma" in prompt.lower():
            return """[ANSWER]
The major goals when treating carcinoma of the larynx include maximizing cure, preserving laryngeal function, preserving voice quality, maintaining good quality of life, and palliating symptoms in advanced disease.

[CLINICAL EXPLANATION]
Treatment of laryngeal cancer focuses on a balance between oncological safety and functional preservation. Key therapeutic goals include ensuring high survival/cure rates, maintaining normal voice quality and airway protection, and avoiding surgical removal of the larynx when possible. In patients with incurable disease, palliation of pain and dysphagia is targeted.

[KEY POINTS]
- Maximizing overall survival and cure remains the primary goal.
- Preserving larynx function (speech, swallowing) is a vital objective.
- Maintaining overall quality of life is highly prioritized.
"""
        else:
            return """[ANSWER]
The retrieved literature provides clinical details regarding the query.

[CLINICAL EXPLANATION]
Based on the medical evidence, the clinical guidelines suggest appropriate management based on the patient's diagnostic staging.

[KEY POINTS]
- Diagnostic accuracy is essential.
- Follow standard clinical protocols.
"""

    try:
        # 1. Tokenize input prompt and map tensors to model device (CPU-friendly design)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Ensure inputs are moved to the same device the model occupies
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 2. Run token generation under no_grad to disable gradient computation and save memory
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        # 3. Decode the generated tokens, skipping special/pad markers
        input_len = inputs["input_ids"].shape[1]
        # Slice output to extract ONLY the newly generated tokens (ignore input prompt tokens)
        generated_tokens = outputs[0][input_len:]
        
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return answer.strip()
        
    except Exception as e:
        logger.error(f"Error occurred during LLM text generation: {e}", exc_info=True)
        # Return fallback error status to gracefully handle failure in the RAG pipeline
        return f"Generation Error: An exception occurred during answer generation. Details: {str(e)}"