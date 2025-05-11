import numpy as np

class Evaluator:
    def __init__(self):
        """Initialize the evaluation module."""
        pass
    
    def evaluate_bias_reduction(self, original_rankings, debiased_rankings):
        """Evaluate how much bias was reduced.
        
        Args:
            original_rankings: List of (resume, score) tuples from original model
            debiased_rankings: List of (resume, score) tuples from debiased model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract scores
        original_scores = np.array([score for _, score in original_rankings])
        debiased_scores = np.array([score for _, score in debiased_rankings])
        
        # Calculate change in rankings
        orig_rank_map = {r: i for i, (r, _) in enumerate(original_rankings)}
        debias_rank_map = {r: i for i, (r, _) in enumerate(debiased_rankings)}
        
        common_resumes = set(orig_rank_map.keys()) & set(debias_rank_map.keys())
        rank_changes = []
        
        for resume in common_resumes:
            orig_rank = orig_rank_map[resume]
            debias_rank = debias_rank_map[resume]
            rank_changes.append(abs(orig_rank - debias_rank))
        
        # Calculate metrics
        metrics = {
            'original_mean_score': np.mean(original_scores),
            'debiased_mean_score': np.mean(debiased_scores),
            'original_std': np.std(original_scores),
            'debiased_std': np.std(debiased_scores),
            'mean_rank_change': np.mean(rank_changes) if rank_changes else 0,
            'max_rank_change': max(rank_changes) if rank_changes else 0
        }
        
        return metrics