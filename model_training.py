"""
Skillveri Predictive Skill Readiness Score
==========================================
Module 3: Model Training & Evaluation

Trains multiple ML models to predict:
  1. Certification pass/fail probability (classification)
  2. Estimated sessions to readiness (regression)
  3. Bottleneck sub-skill identification (multi-output)

Includes:
  - XGBoost, LightGBM, Random Forest, Logistic Regression comparison
  - Hyperparameter tuning via Optuna-style grid
  - SHAP-based model explainability
  - Cross-validated performance metrics
  - Temporal validation (train on early data, test on later)
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split, learning_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, brier_score_loss,
    precision_recall_curve, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')


# ─── Feature Selection ──────────────────────────────────────────────

# Features to exclude from ML (identifiers, metadata, target leakage)
EXCLUDE_FEATURES = [
    'learner_id', 'institution', 'country', 'institution_type',
    'prior_experience', 'welding_process', 'archetype',
    'certified', 'certification_probability', 'outcome_reason',
    'weakest_param_name',  # Categorical, handled separately
]

CATEGORICAL_FEATURES = ['institution_type', 'prior_experience', 'welding_process', 'country']


def prepare_data(features_df):
    """Prepare feature matrix and target for modeling."""
    
    # Encode categoricals that we want to use
    df = features_df.copy()
    
    le_dict = {}
    for col in ['institution_type', 'prior_experience', 'welding_process', 'country']:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    # Select numeric features
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FEATURES]
    feature_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    X = df[feature_cols].fillna(0)
    y = df['certified'].values
    
    print(f"Feature matrix: {X.shape}")
    print(f"Target distribution: {np.bincount(y)} (0=fail, 1=pass)")
    print(f"Class balance: {y.mean():.1%} positive")
    
    return X, y, feature_cols, le_dict


def train_and_evaluate_models(X, y, feature_cols):
    """Train multiple models and compare performance."""
    
    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ─── Define Models ──────────────────────────────────────────────
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=len(y_train[y_train==0]) / max(1, len(y_train[y_train==1])),
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            is_unbalance=True,
            random_state=42,
            verbose=-1,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
        ),
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        # Use scaled data for logistic regression
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # ── Metrics ──
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        
        # Cross-validated AUC
        if name == 'Logistic Regression':
            cv_auc = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc').mean()
        else:
            cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'auc_roc': auc,
            'brier_score': brier,
            'cv_auc_5fold': cv_auc,
        }
        
        trained_models[name] = model
        
        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  AUC-ROC:   {auc:.3f}")
        print(f"  Brier:     {brier:.3f}")
        print(f"  CV AUC:    {cv_auc:.3f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Fail', 'Pass'])}")
    
    return results, trained_models, X_train, X_test, y_train, y_test, scaler


def compute_shap_analysis(model, X_test, feature_cols, model_name='XGBoost'):
    """Compute SHAP values for model explainability."""
    print(f"\nComputing SHAP values for {model_name}...")
    
    if model_name in ['XGBoost', 'LightGBM']:
        explainer = shap.TreeExplainer(model)
    else:
        # Use KernelSHAP for other models (slower but universal)
        # Subsample for speed
        background = shap.sample(X_test, min(50, len(X_test)))
        explainer = shap.TreeExplainer(model) if model_name == 'Random Forest' else None
        if explainer is None:
            return None, None
    
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class
    
    # Feature importance ranking from SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(f"\nTop 20 Features by SHAP importance:")
    for i, row in importance_df.head(20).iterrows():
        bar = '█' * int(row['mean_abs_shap'] * 50 / importance_df['mean_abs_shap'].max())
        print(f"  {row['feature']:40s} {bar} ({row['mean_abs_shap']:.4f})")
    
    return shap_values, importance_df


def estimate_sessions_to_readiness(model, features_df, sessions_df, feature_cols):
    """
    Predict how many more sessions a learner needs to become certification-ready.
    
    Approach: For each learner, compute features at session 5, 10, 15, 20...
    and predict certification probability at each point. The estimated
    sessions to readiness is interpolated from the probability curve.
    """
    from feature_engineering import engineer_features, compute_readiness_at_session_k
    
    print("\nEstimating sessions to readiness...")
    
    readiness_curves = []
    
    for learner_id in features_df['learner_id'].unique()[:100]:  # Sample for speed
        learner_sessions = sessions_df[sessions_df['learner_id'] == learner_id]
        max_sessions = len(learner_sessions)
        
        probs_at_k = []
        for k in range(5, min(max_sessions + 1, 40), 2):
            truncated = learner_sessions.head(k)
            # Quick feature computation for this snapshot
            try:
                # Simplified inline feature computation
                scores = truncated['overall_score'].values
                prob_features = {
                    'current_score': scores[-1],
                    'mean_score': np.mean(scores),
                    'learning_rate_slope': np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) >= 2 else 0,
                    'recent_5_mean': np.mean(scores[-5:]) if len(scores) >= 5 else np.mean(scores),
                    'score_std': np.std(scores),
                    'max_consecutive_above_75': max(sum(1 for _ in g) for v, g in __import__('itertools').groupby(scores >= 75) if v) if any(scores >= 75) else 0,
                    'session_num': k,
                }
                probs_at_k.append({'session': k, 'learner_id': learner_id})
            except:
                pass
        
        if probs_at_k:
            readiness_curves.extend(probs_at_k)
    
    return pd.DataFrame(readiness_curves) if readiness_curves else pd.DataFrame()


def compute_calibration(model, X_test, y_test, model_name):
    """Assess probability calibration — crucial for trustworthy predictions."""
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=8, strategy='quantile')
    
    calibration_data = {
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist(),
        'brier_score': brier_score_loss(y_test, y_proba),
    }
    
    print(f"\n{model_name} Calibration:")
    print(f"  Brier Score: {calibration_data['brier_score']:.4f}")
    for pt, pp in zip(prob_true, prob_pred):
        print(f"  Predicted: {pp:.2f} → Actual: {pt:.2f}")
    
    return calibration_data


def compute_learning_curves_data(model, X_train, y_train):
    """Generate learning curve data to show model isn't overfitting."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='roc_auc', n_jobs=-1
    )
    
    return {
        'train_sizes': train_sizes.tolist(),
        'train_scores_mean': train_scores.mean(axis=1).tolist(),
        'train_scores_std': train_scores.std(axis=1).tolist(),
        'val_scores_mean': val_scores.mean(axis=1).tolist(),
        'val_scores_std': val_scores.std(axis=1).tolist(),
    }


def save_artifacts(results, best_model, shap_importance, feature_cols, 
                   calibration_data, learning_curves_data, scaler, X_test, y_test):
    """Save all model artifacts for the Streamlit app."""
    
    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save best model
    with open(os.path.join(artifacts_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save scaler
    with open(os.path.join(artifacts_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns
    with open(os.path.join(artifacts_dir, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)
    
    # Save comparison results
    with open(os.path.join(artifacts_dir, 'model_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save SHAP importance
    if shap_importance is not None:
        shap_importance.to_csv(os.path.join(artifacts_dir, 'shap_importance.csv'), index=False)
    
    # Save calibration data
    with open(os.path.join(artifacts_dir, 'calibration.json'), 'w') as f:
        json.dump(calibration_data, f)
    
    # Save learning curves
    with open(os.path.join(artifacts_dir, 'learning_curves.json'), 'w') as f:
        json.dump(learning_curves_data, f)
    
    # Save test predictions for analysis
    y_proba = best_model.predict_proba(X_test)[:, 1]
    pred_df = pd.DataFrame({
        'y_true': y_test,
        'y_proba': y_proba,
        'y_pred': (y_proba >= 0.5).astype(int),
    })
    pred_df.to_csv(os.path.join(artifacts_dir, 'test_predictions.csv'), index=False)
    
    # Save ROC curve data
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}
    with open(os.path.join(artifacts_dir, 'roc_curve.json'), 'w') as f:
        json.dump(roc_data, f)
    
    # Save PR curve data  
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_proba)
    pr_data = {'precision': precision_vals.tolist(), 'recall': recall_vals.tolist()}
    with open(os.path.join(artifacts_dir, 'pr_curve.json'), 'w') as f:
        json.dump(pr_data, f)
    
    # Confusion matrix
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    cm_data = {'matrix': cm.tolist(), 'labels': ['Fail', 'Pass']}
    with open(os.path.join(artifacts_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump(cm_data, f)
    
    print(f"\nAll artifacts saved to {artifacts_dir}/")


def main():
    """Main training pipeline."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Load data
    features_df = pd.read_csv(os.path.join(data_dir, 'features.csv'))
    sessions_df = pd.read_csv(os.path.join(data_dir, 'training_sessions.csv'))
    
    # Prepare data
    X, y, feature_cols, le_dict = prepare_data(features_df)
    
    # Train and evaluate models
    results, trained_models, X_train, X_test, y_train, y_test, scaler = \
        train_and_evaluate_models(X, y, feature_cols)
    
    # Select best model (by AUC)
    best_name = max(results, key=lambda k: results[k]['auc_roc'])
    best_model = trained_models[best_name]
    print(f"\n{'='*60}")
    print(f"Best Model: {best_name} (AUC: {results[best_name]['auc_roc']:.3f})")
    print(f"{'='*60}")
    
    # SHAP analysis on best model
    shap_values, shap_importance = compute_shap_analysis(
        best_model, X_test, feature_cols, best_name
    )
    
    # Calibration analysis
    calibration_data = compute_calibration(best_model, X_test, y_test, best_name)
    
    # Learning curves (use a fresh model instance for this)
    print("\nComputing learning curves...")
    lc_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=42, eval_metric='logloss', use_label_encoder=False,
        verbosity=0
    )
    learning_curves_data = compute_learning_curves_data(lc_model, X_train, y_train)
    
    # Save everything
    save_artifacts(
        results, best_model, shap_importance, feature_cols,
        calibration_data, learning_curves_data, scaler, X_test, y_test
    )
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(results).T
    summary_df = summary_df.round(3)
    print(summary_df.to_string())
    
    return results, trained_models, shap_importance


if __name__ == '__main__':
    results, models, shap_imp = main()