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
```text
FairHire-AI/
├── data/
│   ├── bias_patterns/          # Predefined bias-indicating patterns
│   └── resumes/                # Original and test resumes
├── models/
│   ├── adversarial.py          # Bias detection logic
│   ├── primary_ai.py           # Resume ranking model
│   └── unlearning.py           # Machine unlearning implementation
├── utils/
│   ├── data_loader.py          # Data loading and preprocessing
│   └── evaluator.py            # Fairness and performance evaluation
├── config.py                   # Configuration and constants
├── main.py                     # Entry point for running the pipeline
└── Report.doc                  # Project report/documentation 

```


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
```bash
python main.py




