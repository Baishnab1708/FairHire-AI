import os
import random

class DataLoader:
    def __init__(self, resumes_dir='data/resumes'):
        """Initialize data loader.
        
        Args:
            resumes_dir: Directory containing resume data
        """
        self.resumes_dir = resumes_dir
        
    def load_resumes(self):
        """Load resumes from data directory.
        
        Returns:
            List of resume texts
        """
        # If directory doesn't exist, generate sample data
        if not os.path.exists(self.resumes_dir):
            return self.generate_sample_resumes()
            
        resumes = []
        for filename in os.listdir(self.resumes_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(self.resumes_dir, filename), 'r') as f:
                    resumes.append(f.read())
        
        return resumes
    
    def generate_sample_resumes(self, n=100):
        """Generate sample resume data for testing.
        
        Args:
            n: Number of resumes to generate
            
        Returns:
            List of sample resumes
        """
        print(f"Generating {n} sample resumes...")
        
        # Components to generate diverse resumes
        education = [
            "Harvard University", "Stanford University", "MIT", 
            "State University", "Community College", "Online University"
        ]
        
        companies = [
            "Google", "Microsoft", "Local Startup", "Small Business",
            "Fortune 500 Company", "Tech Innovators Inc."
        ]
        
        skills = [
            "Python", "Java", "JavaScript", "management", "leadership",
            "communication", "teamwork", "problem-solving", "creativity"
        ]
        
        names = [
            "John Smith", "Jane Doe", "Robert Johnson", "Maria Garcia",
            "Wei Chen", "Aisha Mohammed", "Rahul Patel", "Sarah Jones"
        ]
        
        resumes = []
        
        for i in range(n):
            # Randomly select components
            name = random.choice(names)
            edu = random.sample(education, k=random.randint(1, 2))
            comps = random.sample(companies, k=random.randint(1, 3))
            resume_skills = random.sample(skills, k=random.randint(3, 6))
            
            # Add some intentional bias patterns for demonstration
            if i < n/4:  # 25% of resumes have "elite" markers
                edu = ["Harvard University", "Stanford University"]
                comps = ["Google", "Microsoft"]
            
            # Construct resume
            resume = f"Resume: {name}\n\n"
            resume += "Education:\n"
            for school in edu:
                resume += f"- {school}, {random.randint(2010, 2022)}\n"
                
            resume += "\nExperience:\n"
            for company in comps:
                years = random.randint(1, 5)
                resume += f"- {company}, {years} year{'s' if years > 1 else ''}\n"
                
            resume += "\nSkills:\n"
            resume += ", ".join(resume_skills)
            
            resumes.append(resume)
        
        return resumes
    
    def create_training_data(self, resumes):
        """Create training data with simulated ratings.
        
        Args:
            resumes: List of resume texts
            
        Returns:
            (resumes, labels) tuple for training
        """
        labels = []
        
        for resume in resumes:
            # Simulate biased ratings for demonstration
            score = 0.5  # Base score
            
            # Education bias
            if "Harvard" in resume or "Stanford" in resume:
                score += 0.3
            elif "Community College" in resume:
                score -= 0.2
                
            # Company bias
            if "Google" in resume or "Microsoft" in resume:
                score += 0.2
                
            # Gender bias (subtle)
            if "John" in resume or "Robert" in resume:
                score += 0.1
            
            # Ensure score is between 0 and 1
            score = max(0, min(1, score))
            labels.append(score)
            
        return resumes, labels