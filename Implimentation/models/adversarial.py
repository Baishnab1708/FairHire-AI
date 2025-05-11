import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class AdversarialAI:
    def __init__(self, sensitive_terms=None):
        """Initialize the adversarial AI for bias detection.
        
        Args:
            sensitive_terms: Dictionary mapping categories to lists of sensitive terms
        """
        self.sensitive_terms = sensitive_terms or {
            'gender': ['he', 'she', 'him', 'her', 'male', 'female', 'man', 'woman'],
            'race': ['black', 'white', 'hispanic', 'asian', 'native', 'minority'],
            'age': ['young', 'old', 'senior', 'junior', 'veteran', 'fresh', 'graduate']
        }
        self.vectorizers = {}
        
        # Create a vectorizer for each category
        for category, terms in self.sensitive_terms.items():
            self.vectorizers[category] = CountVectorizer(vocabulary=terms)
    
    def detect_bias(self, ranked_resumes, top_k=10):
        """Detect potential bias in ranked resumes.
        
        Args:
            ranked_resumes: List of (resume, score) tuples from primary AI
            top_k: Number of top resumes to analyze
            
        Returns:
            Dictionary with bias analysis results
        """
        top_resumes = [r[0] for r in ranked_resumes[:top_k]]
        bottom_resumes = [r[0] for r in ranked_resumes[-top_k:]]
        
        bias_analysis = {}
        
        # Check for each type of bias
        for category, vectorizer in self.vectorizers.items():
            # Analyze top resumes
            top_features = vectorizer.transform(top_resumes)
            top_counts = np.sum(top_features.toarray(), axis=0)
            top_term_counts = dict(zip(vectorizer.get_feature_names_out(), top_counts))
            
            # Analyze bottom resumes
            bottom_features = vectorizer.transform(bottom_resumes)
            bottom_counts = np.sum(bottom_features.toarray(), axis=0)
            bottom_term_counts = dict(zip(vectorizer.get_feature_names_out(), bottom_counts))
            
            # Calculate bias score (difference in prevalence)
            bias_score = sum(top_counts) / len(top_resumes) - sum(bottom_counts) / len(bottom_resumes)
            
            bias_analysis[category] = {
                'bias_score': bias_score,
                'top_terms': top_term_counts,
                'bottom_terms': bottom_term_counts
            }
        
        # Determine if there's significant bias
        significant_bias = any(abs(analysis['bias_score']) > 0.2 for analysis in bias_analysis.values())
        
        return {
            'has_significant_bias': significant_bias,
            'categories': bias_analysis
        }
    

class BiasDetector:
    def __init__(self, threshold=0.2):
        """Initialize bias detector component.
        
        Args:
            threshold: Threshold above which bias is considered significant
        """
        self.threshold = threshold
        
    def analyze_rankings(self, ranked_resumes, feature_importance):
        """Analyze rankings for specific biasing features.
        
        Args:
            ranked_resumes: List of (resume, score) tuples
            feature_importance: Dictionary of feature -> importance scores
            
        Returns:
            Dictionary with detailed bias analysis
        """
        # Known bias-indicating terms grouped by category
        bias_indicators = {
            'gender_bias': ['male', 'female', 'man', 'woman', 'he', 'she'],
            'education_bias': ['harvard', 'stanford', 'ivy', 'elite', 'top-tier'],
            'age_bias': ['veteran', 'senior', 'junior', 'young', 'experience'],
            'name_bias': ['foreign', 'ethnic', 'traditional', 'western']
        }
        
        bias_analysis = {}
        triggered_categories = []
        
        # Analyze feature importance for each bias category
        for category, terms in bias_indicators.items():
            category_score = 0
            category_terms = {}
            
            # Sum up importance scores for bias-related terms
            for term in terms:
                if term in feature_importance:
                    score = feature_importance[term]
                    category_score += abs(score)
                    category_terms[term] = score
            
            bias_analysis[category] = {
                'score': category_score,
                'significant': category_score > self.threshold,
                'terms': category_terms
            }
            
            if bias_analysis[category]['significant']:
                triggered_categories.append(category)
        
        return {
            'triggered': len(triggered_categories) > 0,
            'categories': triggered_categories,
            'analysis': bias_analysis
        }