# FairHire-AI
# FairHire AI 🛡️: Privacy-Preserving and Fair Resume Screening System

## 🔍 Overview
FairHire AI is a machine learning system designed to detect, reduce, and mitigate biases in resume screening models. It ensures that AI-driven hiring does not unfairly favor or discriminate based on sensitive features such as gender, ethnicity, or age.

The system performs:
- **Bias detection** across labeled training data.
- **Machine unlearning** to remove biased feature influence.
- **Debiased resume ranking** using a retrained model.
- **Ethical evaluation** to assess fairness improvement post-unlearning.

---

## 🚀 Features
- ✅ Identifies and removes biased terms in resumes.
- 🧠 Retrains ML models without biased features (Machine Unlearning).
- 📊 Displays fair ranking of resumes.
- 🔒 Built with privacy and fairness as core principles.

---

## 🗃️ Project Structure
.
├── data/
│ ├── bias_patterns/
│ └── resumes/
├── models/
│ ├── adversarial.py
│ ├── primary_ai.py
│ └── unlearning.py
├── utils/
│ ├── data_loader.py
│ └── evaluator.py
├── config.py
├── main.py
└── Report.doc


---

## ⚙️ How It Works

1. **Bias Detection**  
   `adversarial.py` analyzes resumes for biased patterns using predefined sensitive feature categories.

2. **Machine Unlearning**  
   `unlearning.py` removes biased terms using TF-IDF vectorization and retrains the model using Logistic Regression.

3. **Resume Ranking**  
   `primary_ai.py` ranks the test resumes post-bias removal for fair evaluation.

4. **Evaluation**  
   `evaluator.py` quantifies improvement in fairness and performance.

---

## 💻 Run Locally


### Prerequisites
```bash
pip install -r requirements.txt



