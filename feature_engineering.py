"""
Skillveri Predictive Skill Readiness Score
==========================================
Module 2: Feature Engineering

Extracts ML-ready features from raw session data. This is the core
intellectual contribution — domain-specific feature engineering that
captures the nuances of psychomotor skill acquisition.

Feature Categories:
  1. Trajectory Features — learning curve shape, acceleration, plateau detection
  2. Consistency Features — score variance, session-to-session stability
  3. Sub-Skill Features — per-parameter strengths, weaknesses, correlations
  4. Practice Pattern Features — frequency, gaps, session duration trends
  5. Difficulty Progression Features — how scores change with harder positions
  6. Relative Features — performance vs cohort/institution averages
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')


def compute_trajectory_features(group):
    """
    Extract features that capture the SHAPE of the learning curve.
    Grounded in skill acquisition theory (power law of practice).
    """
    scores = group['overall_score'].values
    sessions = group['session_number'].values
    n = len(scores)
    
    features = {}
    
    # ── Basic trajectory stats ──
    features['current_score'] = scores[-1]
    features['max_score_achieved'] = np.max(scores)
    features['mean_score'] = np.mean(scores)
    features['median_score'] = np.median(scores)
    features['score_std'] = np.std(scores)
    features['total_sessions'] = n
    
    # ── Learning rate (slope of linear fit) ──
    if n >= 3:
        slope, intercept, r_value, _, _ = stats.linregress(sessions, scores)
        features['learning_rate_slope'] = slope
        features['learning_rate_r2'] = r_value ** 2
        features['projected_score_30'] = intercept + slope * 30  # Projected score at session 30
    else:
        features['learning_rate_slope'] = 0
        features['learning_rate_r2'] = 0
        features['projected_score_30'] = scores[-1]
    
    # ── Recent trajectory (last 5 sessions vs first 5) ──
    if n >= 10:
        first5_mean = np.mean(scores[:5])
        last5_mean = np.mean(scores[-5:])
        features['improvement_first_to_last5'] = last5_mean - first5_mean
        features['recent_5_mean'] = last5_mean
        features['recent_5_std'] = np.std(scores[-5:])
        
        # Recent momentum (last 3 sessions slope)
        recent_slope, _, _, _, _ = stats.linregress(range(5), scores[-5:])
        features['recent_momentum'] = recent_slope
    else:
        features['improvement_first_to_last5'] = scores[-1] - scores[0]
        features['recent_5_mean'] = np.mean(scores[-min(5, n):])
        features['recent_5_std'] = np.std(scores[-min(5, n):]) if n > 1 else 0
        features['recent_momentum'] = 0
    
    # ── Plateau detection ──
    # A learner has plateaued if the last N sessions show < threshold improvement
    if n >= 8:
        last8_slope, _, _, _, _ = stats.linregress(range(8), scores[-8:])
        features['plateau_detected'] = 1 if abs(last8_slope) < 0.3 else 0
        features['plateau_score'] = np.mean(scores[-8:]) if features['plateau_detected'] else 0
    else:
        features['plateau_detected'] = 0
        features['plateau_score'] = 0
    
    # ── Learning acceleration (2nd derivative) ──
    if n >= 6:
        # Smooth the curve first
        if n >= 7:
            smoothed = savgol_filter(scores, min(7, n if n % 2 == 1 else n-1), 2)
        else:
            smoothed = scores
        first_deriv = np.diff(smoothed)
        second_deriv = np.diff(first_deriv)
        features['learning_acceleration'] = np.mean(second_deriv[-3:]) if len(second_deriv) >= 3 else 0
        features['max_improvement_rate'] = np.max(first_deriv)
        features['session_of_max_improvement'] = np.argmax(first_deriv) + 1
    else:
        features['learning_acceleration'] = 0
        features['max_improvement_rate'] = 0
        features['session_of_max_improvement'] = 1
    
    # ── Score above threshold counts ──
    features['sessions_above_75'] = np.sum(scores >= 75)
    features['sessions_above_80'] = np.sum(scores >= 80)
    features['pct_sessions_above_75'] = np.mean(scores >= 75)
    
    # ── Consecutive high-score streaks ──
    max_streak_75 = 0
    current_streak = 0
    for s in scores:
        if s >= 75:
            current_streak += 1
            max_streak_75 = max(max_streak_75, current_streak)
        else:
            current_streak = 0
    features['max_consecutive_above_75'] = max_streak_75
    
    # Current streak (ending at last session)
    current_streak_end = 0
    for s in reversed(scores):
        if s >= 75:
            current_streak_end += 1
        else:
            break
    features['current_streak_above_75'] = current_streak_end
    
    return features


def compute_consistency_features(group):
    """
    Consistency is crucial for certification — a welder who scores 85 one day
    and 55 the next is not ready, even if their average is 70.
    """
    scores = group['overall_score'].values
    n = len(scores)
    features = {}
    
    # ── Score variability ──
    features['score_cv'] = np.std(scores) / (np.mean(scores) + 1e-8)  # Coefficient of variation
    features['score_iqr'] = np.percentile(scores, 75) - np.percentile(scores, 25)
    features['score_range'] = np.max(scores) - np.min(scores)
    
    # ── Session-to-session consistency ──
    if n >= 2:
        diffs = np.abs(np.diff(scores))
        features['mean_session_diff'] = np.mean(diffs)
        features['max_session_drop'] = np.max(np.diff(scores) * -1)  # Largest single-session drop
        features['pct_sessions_improved'] = np.mean(np.diff(scores) > 0)
    else:
        features['mean_session_diff'] = 0
        features['max_session_drop'] = 0
        features['pct_sessions_improved'] = 0
    
    # ── Recent consistency (more important than historical) ──
    if n >= 5:
        recent = scores[-5:]
        features['recent_consistency'] = 1 - (np.std(recent) / (np.mean(recent) + 1e-8))
        features['recent_min'] = np.min(recent)
    else:
        features['recent_consistency'] = 0.5
        features['recent_min'] = np.min(scores)
    
    # ── Regression detection (are they getting worse?) ──
    if n >= 5:
        mid_point = n // 2
        first_half_std = np.std(scores[:mid_point])
        second_half_std = np.std(scores[mid_point:])
        features['variability_trend'] = second_half_std - first_half_std  # Negative = more consistent
    else:
        features['variability_trend'] = 0
    
    return features


def compute_subskill_features(group):
    """
    Per-parameter analysis — identifies specific weaknesses that might
    prevent certification even if overall score is acceptable.
    """
    params = ['score_work_angle', 'score_travel_angle', 'score_travel_speed',
              'score_contact_tip_distance', 'score_arc_length', 'score_bead_quality']
    
    features = {}
    
    # Latest session parameter scores
    last_session = group.iloc[-1]
    param_scores_latest = [last_session[p] for p in params]
    
    features['weakest_param_score'] = min(param_scores_latest)
    features['strongest_param_score'] = max(param_scores_latest)
    features['param_score_spread'] = max(param_scores_latest) - min(param_scores_latest)
    features['num_params_below_70'] = sum(1 for s in param_scores_latest if s < 70)
    features['num_params_above_80'] = sum(1 for s in param_scores_latest if s >= 80)
    
    # Identify the weakest parameter
    param_names_short = ['work_angle', 'travel_angle', 'travel_speed', 'ctwd', 'arc_length', 'bead_quality']
    weakest_idx = np.argmin(param_scores_latest)
    features['weakest_param_name'] = param_names_short[weakest_idx]
    
    # Average parameter scores (across all sessions)
    for p, name in zip(params, param_names_short):
        features[f'avg_{name}'] = group[p].mean()
        if len(group) >= 5:
            features[f'recent_{name}'] = group[p].tail(5).mean()
            # Per-param improvement
            features[f'improvement_{name}'] = group[p].tail(5).mean() - group[p].head(5).mean()
        else:
            features[f'recent_{name}'] = group[p].mean()
            features[f'improvement_{name}'] = 0
    
    # ── Parameter correlation with overall score ──
    # High correlation means this param drives overall performance
    if len(group) >= 5:
        for p, name in zip(params, param_names_short):
            corr = group[p].corr(group['overall_score'])
            features[f'corr_{name}_overall'] = corr if not np.isnan(corr) else 0
    
    return features


def compute_practice_pattern_features(group):
    """
    Practice patterns significantly affect skill acquisition.
    Frequency, spacing, and session duration all matter.
    """
    features = {}
    
    n = len(group)
    features['total_practice_minutes'] = group['total_practice_minutes'].iloc[-1]
    features['avg_session_duration'] = group['session_duration_min'].mean()
    features['total_exercises'] = group['num_exercises'].sum()
    features['days_since_enrollment'] = group['days_since_enrollment'].iloc[-1]
    
    # Practice frequency
    if features['days_since_enrollment'] > 0:
        features['sessions_per_week'] = n / (features['days_since_enrollment'] / 7)
    else:
        features['sessions_per_week'] = 0
    
    # Session gaps
    gaps = group['session_gap_days'].values[1:] if n > 1 else [0]
    features['avg_gap_days'] = np.mean(gaps)
    features['max_gap_days'] = np.max(gaps) if len(gaps) > 0 else 0
    features['gap_std'] = np.std(gaps) if len(gaps) > 1 else 0
    features['pct_gaps_over_5days'] = np.mean(np.array(gaps) > 5) if len(gaps) > 0 else 0
    
    # Practice consistency (entropy of gaps — lower = more regular)
    if len(gaps) > 2:
        gap_counts = np.bincount(np.clip(gaps, 0, 14))
        gap_probs = gap_counts / gap_counts.sum()
        gap_probs = gap_probs[gap_probs > 0]
        features['practice_regularity'] = -np.sum(gap_probs * np.log2(gap_probs))
    else:
        features['practice_regularity'] = 0
    
    # Consecutive practice days
    features['max_consecutive_days'] = group['consecutive_practice_days'].max()
    features['current_consecutive_days'] = group['consecutive_practice_days'].iloc[-1]
    
    # Duration trend (are sessions getting longer or shorter?)
    if n >= 5:
        dur_slope, _, _, _, _ = stats.linregress(range(n), group['session_duration_min'].values)
        features['duration_trend'] = dur_slope
    else:
        features['duration_trend'] = 0
    
    return features


def compute_difficulty_features(group):
    """
    How well does the learner perform as difficulty increases?
    This is critical for certification readiness — certification
    tests include multiple positions and processes.
    """
    features = {}
    
    # Performance by position difficulty
    for pos in ['1G_Flat', '2G_Horizontal', '3G_Vertical', '4G_Overhead']:
        pos_data = group[group['welding_position'] == pos]
        if len(pos_data) > 0:
            features[f'avg_score_{pos}'] = pos_data['overall_score'].mean()
            features[f'count_{pos}'] = len(pos_data)
        else:
            features[f'avg_score_{pos}'] = 0
            features[f'count_{pos}'] = 0
    
    # Difficulty gradient (how much does score drop per unit difficulty?)
    difficulties = group['position_difficulty'].values
    scores = group['overall_score'].values
    if len(np.unique(difficulties)) >= 2:
        diff_slope, _, _, _, _ = stats.linregress(difficulties, scores)
        features['difficulty_sensitivity'] = diff_slope  # Negative = score drops with difficulty
    else:
        features['difficulty_sensitivity'] = 0
    
    # Highest difficulty attempted
    features['max_difficulty_attempted'] = group['position_difficulty'].max()
    
    # Score at max difficulty
    max_diff_data = group[group['position_difficulty'] == features['max_difficulty_attempted']]
    features['score_at_max_difficulty'] = max_diff_data['overall_score'].mean() if len(max_diff_data) > 0 else 0
    
    # Number of unique positions practiced
    features['num_positions_practiced'] = group['welding_position'].nunique()
    
    return features


def compute_relative_features(group, institution_stats, global_stats):
    """
    Performance relative to peers — are they above or below
    the institutional and global averages?
    """
    features = {}
    
    inst = group['institution'].iloc[0]
    current_score = group['overall_score'].iloc[-1]
    mean_score = group['overall_score'].mean()
    
    if inst in institution_stats.index:
        inst_mean = institution_stats.loc[inst, 'mean']
        inst_std = institution_stats.loc[inst, 'std']
        features['score_vs_institution'] = current_score - inst_mean
        features['zscore_vs_institution'] = (current_score - inst_mean) / (inst_std + 1e-8)
    else:
        features['score_vs_institution'] = 0
        features['zscore_vs_institution'] = 0
    
    features['score_vs_global'] = current_score - global_stats['mean']
    features['zscore_vs_global'] = (current_score - global_stats['mean']) / (global_stats['std'] + 1e-8)
    
    return features


def engineer_features(sessions_df, outcomes_df=None):
    """
    Main feature engineering pipeline.
    Returns a feature matrix ready for ML modeling.
    """
    print("Engineering features...")
    
    # Compute global and institutional statistics
    latest_scores = sessions_df.groupby('learner_id')['overall_score'].last()
    global_stats = {'mean': latest_scores.mean(), 'std': latest_scores.std()}
    
    institution_stats = sessions_df.groupby('institution')['overall_score'].agg(['mean', 'std'])
    
    all_features = []
    
    for learner_id, group in sessions_df.groupby('learner_id'):
        group = group.sort_values('session_number')
        
        features = {'learner_id': learner_id}
        
        # Metadata (non-ML features, kept for analysis)
        features['institution'] = group['institution'].iloc[0]
        features['country'] = group['country'].iloc[0]
        features['institution_type'] = group['institution_type'].iloc[0]
        features['age'] = group['age'].iloc[0]
        features['prior_experience'] = group['prior_experience'].iloc[0]
        features['welding_process'] = group['welding_process'].iloc[0]
        features['archetype'] = group['archetype'].iloc[0]
        
        # Compute all feature groups
        features.update(compute_trajectory_features(group))
        features.update(compute_consistency_features(group))
        features.update(compute_subskill_features(group))
        features.update(compute_practice_pattern_features(group))
        features.update(compute_difficulty_features(group))
        features.update(compute_relative_features(group, institution_stats, global_stats))
        
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # Merge certification outcomes if provided
    if outcomes_df is not None:
        features_df = features_df.merge(
            outcomes_df[['learner_id', 'certified', 'certification_probability', 'outcome_reason']],
            on='learner_id',
            how='left'
        )
    
    print(f"Engineered {features_df.shape[1]} features for {len(features_df)} learners")
    
    return features_df


def compute_readiness_at_session_k(sessions_df, k):
    """
    Compute features using only the first k sessions of each learner.
    This enables the "predict readiness at any point" use case.
    Used to create training data for the temporal prediction model.
    """
    truncated = sessions_df.groupby('learner_id').apply(
        lambda g: g.sort_values('session_number').head(k)
    ).reset_index(drop=True)
    
    # Only include learners who have at least k sessions
    valid_learners = truncated.groupby('learner_id').size()
    valid_learners = valid_learners[valid_learners >= k].index
    truncated = truncated[truncated['learner_id'].isin(valid_learners)]
    
    return truncated


if __name__ == '__main__':
    import os
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    sessions_df = pd.read_csv(os.path.join(data_dir, 'training_sessions.csv'))
    outcomes_df = pd.read_csv(os.path.join(data_dir, 'certification_outcomes.csv'))
    
    features_df = engineer_features(sessions_df, outcomes_df)
    features_df.to_csv(os.path.join(data_dir, 'features.csv'), index=False)
    
    print(f"\nFeature matrix saved: {features_df.shape}")
    print(f"\nFeature categories:")
    print(f"  Trajectory: 18 features")
    print(f"  Consistency: 10 features")
    print(f"  Sub-skill: 30+ features")
    print(f"  Practice: 14 features")
    print(f"  Difficulty: 12 features")
    print(f"  Relative: 4 features")
    print(f"\nCertification distribution:")
    print(features_df['certified'].value_counts())
    print(f"\nSample features for one learner:")
    sample = features_df.iloc[0]
    for col in ['current_score', 'learning_rate_slope', 'recent_momentum', 
                'score_cv', 'weakest_param_score', 'sessions_per_week',
                'max_consecutive_above_75', 'difficulty_sensitivity']:
        print(f"  {col}: {sample[col]:.3f}")