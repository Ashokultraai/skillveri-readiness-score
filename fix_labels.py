"""
Run this AFTER data_generator.py and feature_engineering.py
but BEFORE model_training.py

Fixes certification labels to correlate strongly with features.
"""
import pandas as pd
import numpy as np
import os

def fix_labels():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    features = pd.read_csv(os.path.join(data_dir, 'features.csv'))

    def recompute_cert(row):
        score = 0
        if row['current_score'] >= 78: score += 2
        elif row['current_score'] >= 70: score += 1
        if row['recent_5_mean'] >= 76: score += 2
        elif row['recent_5_mean'] >= 68: score += 1
        if row['max_consecutive_above_75'] >= 3: score += 2
        elif row['max_consecutive_above_75'] >= 2: score += 1
        if row['weakest_param_score'] >= 70: score += 2
        elif row['weakest_param_score'] >= 60: score += 1
        if row['score_cv'] < 0.08: score += 1
        if row['recent_momentum'] > 0: score += 1
        if row['sessions_per_week'] >= 2.5: score += 1
        noise = np.random.normal(0, 0.5)
        total = score + noise
        if total >= 8: return 1, 0.92
        elif total >= 6: return 1, 0.75
        elif total >= 5: return (1 if np.random.random() < 0.5 else 0), 0.50
        elif total >= 3: return 0, 0.25
        else: return 0, 0.08

    np.random.seed(42)
    results = features.apply(recompute_cert, axis=1, result_type='expand')
    features['certified'] = results[0].astype(int)
    features['certification_probability'] = results[1]
    print(f'New certification rate: {features["certified"].mean():.1%}')
    features.to_csv(os.path.join(data_dir, 'features.csv'), index=False)

    outcomes = pd.read_csv(os.path.join(data_dir, 'certification_outcomes.csv'))
    feat_indexed = features.set_index('learner_id')
    outcomes['certified'] = feat_indexed['certified'].reindex(outcomes['learner_id']).values
    outcomes['certification_probability'] = feat_indexed['certification_probability'].reindex(outcomes['learner_id']).values
    outcomes.to_csv(os.path.join(data_dir, 'certification_outcomes.csv'), index=False)
    print('Labels fixed and saved!')

if __name__ == '__main__':
    fix_labels()