import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge  # Changed from LogisticRegression to Ridge

class PrimaryAI:
    def __init__(self):
        """Initialize the primary AI for resume parsing and ranking."""
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = Ridge()  # Ridge regression handles continuous values
        self.is_trained = False
        
    def train(self, resumes, labels):
        """Train the model on resume data.
        
        Args:
            resumes: List of resume text
            labels: List of corresponding ratings (0-1)
        """
        # Transform text to feature vectors
        X = self.vectorizer.fit_transform(resumes)
        # Train model
        self.model.fit(X, labels)
        self.is_trained = True
        print("Primary AI trained on", len(resumes), "resumes")
        
    def rank_resumes(self, resumes):
        """Rank a list of resumes.
        
        Args:
            resumes: List of resume text
            
        Returns:
            List of (resume, score) tuples sorted by score
        """
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")
            
        # Transform resumes to feature vectors
        X = self.vectorizer.transform(resumes)
        
        # Get predicted scores
        scores = self.model.predict(X)
        
        # Ensure scores are between 0 and 1
        scores = np.clip(scores, 0, 1)
        
        # Create resume-score pairs and sort by score
        ranked_resumes = list(zip(resumes, scores))
        ranked_resumes.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_resumes
    
    def get_feature_importance(self):
        """Get important features for ranking decisions.
        
        Returns:
            Dictionary of feature -> importance score
        """
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")
            
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_
        
        return dict(zip(feature_names, coefficients))