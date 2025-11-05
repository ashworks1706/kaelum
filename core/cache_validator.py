import json
import time
from pathlib import Path
from typing import Dict, Optional


class CacheValidator:
    """LLM-powered semantic cache validation.
    
    Uses reasoning LLM to validate if a cached answer would correctly
    satisfy a new query. Collects validation data for fine-tuning.
    """
    
    def __init__(self, llm_client=None, validation_dir: str = ".kaelum/cache_validation"):
        self.llm_client = llm_client
        self.validation_dir = Path(validation_dir)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.validation_log = self.validation_dir / "validation_log.jsonl"
        
    def validate_cache_match(
        self, 
        new_query: str, 
        cached_query: str, 
        cached_answer: str
    ) -> Dict:
        """Validate if cached answer satisfies new query using LLM.
        
        Args:
            new_query: The new incoming query
            cached_query: The original cached query
            cached_answer: The cached answer
            
        Returns:
            Dict with 'valid' (bool), 'confidence' (float), 'reason' (str)
        """
        if not self.llm_client:
            # Fallback: accept if queries are similar enough
            return {
                'valid': True,
                'confidence': 0.7,
                'reason': 'LLM validator not available, accepted based on similarity'
            }
        
        try:
            prompt = f"""Analyze if a cached answer can be reused for a new query.

CACHED QUERY: {cached_query}
CACHED ANSWER: {cached_answer}

NEW QUERY: {new_query}

Question: Would the cached answer FULLY and CORRECTLY satisfy the new query?

Consider:
- Does the cached answer directly answer what the new query asks?
- Are there any constraints, conditions, or specifics in the new query that the cached answer doesn't address?
- Would using this cached answer be misleading or incorrect?

Respond in JSON format:
{{
    "valid": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""

            response = self.llm_client(prompt, max_tokens=200, temperature=0.1)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                if 'valid' not in result:
                    result['valid'] = False
                if 'confidence' not in result:
                    result['confidence'] = 0.5
                if 'reason' not in result:
                    result['reason'] = 'No reason provided'
            except json.JSONDecodeError:
                # Fallback parsing
                result = {
                    'valid': 'true' in response.lower() and 'false' not in response.lower(),
                    'confidence': 0.6,
                    'reason': 'Parsed from non-JSON response'
                }
            
            # Log validation decision
            self._log_validation(new_query, cached_query, cached_answer, result)
            
            return result
            
        except Exception as e:
            # Log error and fail safe (reject cache)
            return {
                'valid': False,
                'confidence': 0.0,
                'reason': f'Validation error: {str(e)}'
            }
    
    def _log_validation(
        self, 
        new_query: str, 
        cached_query: str, 
        cached_answer: str, 
        validation_result: Dict
    ):
        """Log validation decision for fine-tuning."""
        try:
            log_entry = {
                'timestamp': time.time(),
                'new_query': new_query,
                'cached_query': cached_query,
                'cached_answer': cached_answer,
                'validation_result': validation_result
            }
            
            with open(self.validation_log, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            # Don't fail validation if logging fails
            print(f"Warning: Failed to log validation: {e}")
    
    def export_training_data(self, output_file: str = None):
        """Export validation logs as fine-tuning training data.
        
        Args:
            output_file: Path to output JSONL file (default: auto-generated)
            
        Returns:
            str: Path to the output file
        """
        if not self.validation_log.exists():
            print("No validation log found")
            return None
        
        if output_file is None:
            import time
            timestamp = int(time.time())
            output_file = str(self.validation_dir / f"training_data_{timestamp}.jsonl")
        
        with open(self.validation_log, 'r') as f_in:
            with open(output_file, 'w') as f_out:
                for line in f_in:
                    try:
                        entry = json.loads(line)
                        
                        # Format as instruction-following training data
                        training_example = {
                            'instruction': 'Analyze if a cached answer can be reused for a new query.',
                            'input': f"""CACHED QUERY: {entry['cached_query']}
CACHED ANSWER: {entry['cached_answer']}

NEW QUERY: {entry['new_query']}

Question: Would the cached answer FULLY and CORRECTLY satisfy the new query?""",
                            'output': json.dumps(entry['validation_result'])
                        }
                        
                        f_out.write(json.dumps(training_example) + '\n')
                    except Exception as e:
                        print(f"Warning: Skipped malformed log entry: {e}")
        
        print(f"Exported training data to {output_file}")
        return output_file
    
    def get_validation_stats(self) -> Dict:
        """Get statistics about validation decisions.
        
        Returns:
            Dict with total, valid, invalid counts and avg confidence
        """
        if not self.validation_log.exists():
            return {
                'total': 0,
                'valid': 0,
                'invalid': 0,
                'avg_confidence': 0.0,
                'rejection_rate': 0.0
            }
        
        total = 0
        valid = 0
        invalid = 0
        confidence_sum = 0.0
        
        with open(self.validation_log, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    result = entry.get('validation_result', {})
                    total += 1
                    if result.get('valid', False):
                        valid += 1
                    else:
                        invalid += 1
                    confidence_sum += result.get('confidence', 0.0)
                except Exception:
                    continue
        
        avg_confidence = confidence_sum / total if total > 0 else 0.0
        rejection_rate = invalid / total if total > 0 else 0.0
        
        return {
            'total': total,
            'valid': valid,
            'invalid': invalid,
            'avg_confidence': avg_confidence,
            'rejection_rate': rejection_rate
        }
