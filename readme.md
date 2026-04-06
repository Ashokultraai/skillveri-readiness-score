# 🎯 Skillveri Predictive Skill Readiness Score

**An AI/ML system that predicts when a VR training simulator learner will be certification-ready.**

Built for the **Skillveri AI Engineer** interview — demonstrates end-to-end ML engineering from synthetic data generation to production-ready Streamlit dashboard.

---

## The Problem

Training managers at Skillveri's customers (Hyundai, Maruti, Indian Railways) need to answer:
> *"When will this batch of 50 welders be ready for the shop floor?"*

Currently, Skillveri Insights shows descriptive analytics (progress charts, scores). This project adds **predictive analytics** — forecasting certification readiness, estimating sessions-to-competency, and identifying specific bottleneck sub-skills.

## The Solution

An XGBoost classification model trained on 95 engineered features from 16,000+ training sessions across 600 learners. Achieves **0.981 AUC-ROC** and **0.990 5-fold CV AUC**.

### Key Capabilities
1. **Certification Readiness Prediction** — Pass/fail probability with calibrated confidence
2. **Sessions-to-Readiness Estimation** — "This learner needs ~8 more sessions"
3. **Bottleneck Sub-Skill Identification** — "Focus on work_angle in 3G position"
4. **Batch Analytics** — Training manager dashboard for cohort-level insights
5. **SHAP Explainability** — Full transparency into why the model makes each prediction

---

## Project Structure

```
skillveri-readiness-score/
├── app.py                          # Streamlit dashboard (6 pages)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── src/
│   ├── data_generator.py           # Synthetic data generation (logistic learning curves)
│   ├── feature_engineering.py      # 95 features across 6 categories
│   └── model_training.py           # 4-model comparison + SHAP + calibration
├── data/
│   ├── training_sessions.csv       # 16,424 sessions × 27 columns
│   ├── learner_profiles.csv        # 600 learner profiles
│   ├── certification_outcomes.csv  # Ground truth labels
│   └── features.csv                # Engineered feature matrix (600 × 102)
└── models/
    ├── best_model.pkl              # Trained XGBoost model
    ├── model_comparison.json       # 4-model metrics comparison
    ├── shap_importance.csv         # SHAP feature importance ranking
    ├── roc_curve.json              # ROC curve data
    ├── pr_curve.json               # Precision-Recall curve data
    ├── confusion_matrix.json       # Confusion matrix
    ├── learning_curves.json        # Train/val learning curves
    └── calibration.json            # Probability calibration data
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python -m src.data_generator

# Engineer features
python -m src.feature_engineering

# Train models
python -m src.model_training

# Launch dashboard
streamlit run app.py
```

## Technical Highlights

### Data Generation
- Logistic growth learning curves grounded in Fitts & Posner (1967) psychomotor theory
- 5 learner archetypes with realistic variance
- Correlated parameter scores via Cholesky decomposition
- Position difficulty scaling (1G-6G) matching real welding complexity

### Feature Engineering (95 Features)
| Category | Count | Key Features |
|----------|-------|-------------|
| Trajectory | 18 | Learning rate slope, acceleration, plateau detection |
| Consistency | 10 | Score CV, session-to-session stability, regression |
| Sub-Skill | 30+ | Per-parameter scores, improvements, correlations |
| Practice Pattern | 14 | Frequency, gaps, regularity entropy |
| Difficulty | 12 | Position-specific scores, difficulty sensitivity |
| Relative | 4 | Z-scores vs institution and global means |

### Model Results
| Model | AUC-ROC | F1 | CV AUC | Brier |
|-------|---------|----|----|-------|
| **XGBoost** | **0.981** | **0.875** | **0.990** | **0.036** |
| LightGBM | 0.979 | 0.848 | 0.984 | 0.040 |
| Random Forest | 0.971 | 0.824 | 0.987 | 0.044 |
| Logistic Reg. | 0.977 | 0.743 | 0.978 | 0.054 |

---

## Business Impact for Skillveri

- **For Sales:** "Our AI predicts your batch will be job-ready in 3 weeks" → Quantified value proposition
- **For Customers:** Proactive intervention for struggling learners → Higher certification rates
- **For Product:** Differentiator vs competitors (Interplay SAM, Lincoln VRTEX, Soldamatic) → None has predictive readiness
- **For Training Managers:** Batch-level analytics with bottleneck identification → Data-driven curriculum design

---

*Built by [Your Name] for the Skillveri AI Engineer interview, April 2026*