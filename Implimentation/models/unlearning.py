import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class MachineUnlearning:
    def __init__(self):
        """Initialize the machine unlearning module."""
        self.biased_features = set()
        
    def identify_biased_features(self, bias_analysis):
        """Identify features that contribute to bias.
        
        Args:
            bias_analysis: Results from bias detector
            
        Returns:
            Set of biased features to unlearn
        """
        biased_features = set()
        
        # Extract terms from triggered categories
        if bias_analysis['triggered']:
            for category in bias_analysis['categories']:
                category_data = bias_analysis['analysis'][category]
                for term, score in category_data['terms'].items():
                    biased_features.add(term)
        
        self.biased_features.update(biased_features)
        return biased_features
    
    def unlearn_bias(self, primary_ai, resumes, labels):
        """Retrain model to unlearn biased patterns.
        
        Args:
            primary_ai: The primary AI model
            resumes: Training resumes
            labels: Training labels
            
        Returns:
            Updated primary AI
        """
        if not self.biased_features:
            return primary_ai
        
        print(f"Unlearning {len(self.biased_features)} biased features")
        
        # Create a new vectorizer that ignores biased terms
        new_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(self.biased_features)
        )
        
        # Transform resumes with new vectorizer
        X = new_vectorizer.fit_transform(resumes)
        
        # Train a new model
        new_model = LogisticRegression()
        new_model.fit(X, labels)
        
        # Update the primary AI
        primary_ai.vectorizer = new_vectorizer
        primary_ai.model = new_model
        primary_ai.is_trained = True
        
        return primary_ai