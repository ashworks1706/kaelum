from typing import Dict, List
from transformers import pipeline


class CompletenessDetector:
    
    def __init__(self):
        self.nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli")
    
    def is_complete(self, query: str, response: str, context: List[str] = None) -> Dict:
        context = context or []
        
        try:
            result = self.nli_pipeline(
                f"{response}",
                f"hypothesis: This fully answers the question: {query}"
            )
            
            nli_score = result[0]['score'] if result[0]['label'] == 'ENTAILMENT' else 0.0
        except:
            nli_score = 0.0
        
        question_coverage = self._check_question_coverage(query, response)
        
        has_dangling = self._check_dangling_references(response, context)
        dangling_penalty = 0.2 if has_dangling else 0.0
        
        length_adequacy = self._assess_length_adequacy(query, response)
        
        completeness_score = (
            nli_score * 0.5 +
            question_coverage * 0.3 +
            length_adequacy * 0.2 -
            dangling_penalty
        )
        
        is_complete = completeness_score >= 0.65
        
        missing_aspects = self._identify_missing_aspects(query, response)
        
        return {
            'is_complete': is_complete,
            'confidence': completeness_score,
            'nli_score': nli_score,
            'question_coverage': question_coverage,
            'has_dangling_references': has_dangling,
            'length_adequacy': length_adequacy,
            'missing_aspects': missing_aspects
        }
    
    def _check_question_coverage(self, query: str, response: str) -> float:
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        questions_in_query = []
        for qw in question_words:
            if qw in query_lower:
                questions_in_query.append(qw)
        
        if not questions_in_query:
            return 0.8
        
        coverage = 0.0
        for qw in questions_in_query:
            if qw == 'what' and any(term in response_lower for term in ['is', 'are', 'means', 'refers to']):
                coverage += 1.0
            elif qw == 'why' and any(term in response_lower for term in ['because', 'due to', 'reason', 'since']):
                coverage += 1.0
            elif qw == 'how' and any(term in response_lower for term in ['by', 'through', 'process', 'method', 'steps']):
                coverage += 1.0
            elif qw == 'when' and any(term in response_lower for term in ['at', 'on', 'in', 'during', 'date', 'time']):
                coverage += 1.0
            elif qw == 'where' and any(term in response_lower for term in ['at', 'in', 'location', 'place']):
                coverage += 1.0
            elif qw == 'who' and any(term in response_lower for term in ['person', 'people', 'by', 'name']):
                coverage += 1.0
            else:
                coverage += 0.5
        
        return min(coverage / len(questions_in_query), 1.0)
    
    def _check_dangling_references(self, response: str, context: List[str]) -> bool:
        dangling_indicators = [
            'as mentioned', 'see above', 'refer to', 'as shown',
            'this means', 'that is', 'these are', 'those are'
        ]
        
        response_lower = response.lower()
        
        if not context:
            for indicator in dangling_indicators[:4]:
                if indicator in response_lower:
                    return True
        
        pronoun_starts = ['this ', 'that ', 'these ', 'those ', 'it ', 'they ']
        first_sentence = response.split('.')[0].lower()
        
        if any(first_sentence.startswith(pron) for pron in pronoun_starts) and not context:
            return True
        
        return False
    
    def _assess_length_adequacy(self, query: str, response: str) -> float:
        query_words = len(query.split())
        response_words = len(response.split())
        
        if query_words < 5:
            expected_min = 10
            expected_max = 100
        elif query_words < 15:
            expected_min = 30
            expected_max = 200
        else:
            expected_min = 50
            expected_max = 300
        
        if response_words < expected_min:
            return response_words / expected_min
        elif response_words > expected_max:
            return max(0.8, 1.0 - (response_words - expected_max) / expected_max)
        else:
            return 1.0
    
    def _identify_missing_aspects(self, query: str, response: str) -> List[str]:
        missing = []
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        if 'why' in query_lower and not any(term in response_lower for term in ['because', 'reason', 'due to', 'since']):
            missing.append('explanation of cause/reason')
        
        if 'how' in query_lower and not any(term in response_lower for term in ['step', 'process', 'method', 'by']):
            missing.append('procedural steps or method')
        
        if any(term in query_lower for term in ['compare', 'difference', 'versus', 'vs']) and \
           not any(term in response_lower for term in ['whereas', 'while', 'on the other hand', 'however', 'unlike']):
            missing.append('comparative analysis')
        
        if any(term in query_lower for term in ['advantage', 'benefit', 'pro']) and \
           any(term in query_lower for term in ['disadvantage', 'drawback', 'con']):
            if response_lower.count('advantage') < 1 or response_lower.count('disadvantage') < 1:
                missing.append('both advantages and disadvantages')
        
        return missing
