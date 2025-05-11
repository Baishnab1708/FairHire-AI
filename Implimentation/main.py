from models.primary_ai import PrimaryAI
from models.adversarial import AdversarialAI, BiasDetector
from models.unlearning import MachineUnlearning
from utils.data_loader import DataLoader
from utils.evaluator import Evaluator

def main():
    print("Starting Debiased Resume Ranking System")
    
    # Load data
    data_loader = DataLoader()
    resumes = data_loader.load_resumes()
    resumes, labels = data_loader.create_training_data(resumes)
    
    # Split data for demonstration
    split = int(len(resumes) * 0.8)
    train_resumes, train_labels = resumes[:split], labels[:split]
    test_resumes = resumes[split:]
    
    # Initialize components
    primary_ai = PrimaryAI()
    adversarial_ai = AdversarialAI()
    bias_detector = BiasDetector()
    unlearning = MachineUnlearning()
    evaluator = Evaluator()
    
    # Train primary AI
    print("\n1. Training Primary AI...")
    primary_ai.train(train_resumes, train_labels)
    
    # Get initial rankings
    print("\n2. Ranking resumes...")
    original_rankings = primary_ai.rank_resumes(test_resumes)
    
    # Display top 5 original rankings
    print("\nTop 5 Original Rankings:")
    for i, (resume, score) in enumerate(original_rankings[:5]):
        first_line = resume.split('\n')[0]
        print(f"{i+1}. {first_line} (Score: {score:.2f})")
    
    # Detect bias
    print("\n3. Detecting bias...")
    feature_importance = primary_ai.get_feature_importance()
    adversarial_results = adversarial_ai.detect_bias(original_rankings)
    bias_analysis = bias_detector.analyze_rankings(original_rankings, feature_importance)
    
    # Display bias analysis
    print("\nBias Analysis:")
    print(f"Has significant bias: {adversarial_results['has_significant_bias']}")
    
    for category, data in adversarial_results['categories'].items():
        print(f"- {category}: Bias score {data['bias_score']:.2f}")
    
    if bias_analysis['triggered']:
        print("\nBias categories triggered:")
        for category in bias_analysis['categories']:
            print(f"- {category}")
    
    # Unlearn bias if detected
    if bias_analysis['triggered']:
        print("\n4. Unlearning biased patterns...")
        biased_features = unlearning.identify_biased_features(bias_analysis)
        
        print(f"Identified biased features: {', '.join(biased_features)}")
        
        primary_ai = unlearning.unlearn_bias(primary_ai, train_resumes, train_labels)
        
        # Get debiased rankings
        print("\n5. Generating debiased rankings...")
        debiased_rankings = primary_ai.rank_resumes(test_resumes)
        
        # Display top 5 debiased rankings
        print("\nTop 5 Debiased Rankings:")
        for i, (resume, score) in enumerate(debiased_rankings[:5]):
            first_line = resume.split('\n')[0]
            print(f"{i+1}. {first_line} (Score: {score:.2f})")
        
        # Evaluate improvement
        print("\n6. Evaluating bias reduction...")
        metrics = evaluator.evaluate_bias_reduction(original_rankings, debiased_rankings)
        
        print("\nEvaluation Metrics:")
        print(f"Original mean score: {metrics['original_mean_score']:.2f}")
        print(f"Debiased mean score: {metrics['debiased_mean_score']:.2f}")
        print(f"Mean rank change: {metrics['mean_rank_change']:.2f} positions")
    else:
        print("\nNo significant bias detected. No unlearning necessary.")

if __name__ == "__main__":
    main()