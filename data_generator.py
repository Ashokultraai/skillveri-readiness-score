"""
Skillveri Predictive Skill Readiness Score
==========================================
Module 1: Synthetic Data Generator

Generates realistic welding training session data that mimics
what Skillveri's AURA simulator would produce. Each session tracks:
- Learner demographics & metadata
- Per-session scores across 6 welding parameters
- Time-series learning curves with realistic noise
- Multiple welding positions (1G-6G) and processes (GMAW, GTAW, SMAW)
- Certification pass/fail outcomes

Data Design Philosophy:
  We model learning curves using a logistic growth function with
  individual variation in learning rate, ceiling, and noise. This
  reflects real psychomotor skill acquisition literature where:
  - Initial rapid improvement (steep learning phase)
  - Gradual plateau (diminishing returns)
  - Individual differences in learning speed and ceiling
  - Session-to-session variability (bad days, fatigue)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

np.random.seed(42)

# ─── Domain Constants ───────────────────────────────────────────────
WELDING_PROCESSES = ['GMAW', 'GTAW', 'SMAW']
WELDING_POSITIONS = ['1G_Flat', '2G_Horizontal', '3G_Vertical', '4G_Overhead', '5G_Pipe', '6G_Pipe45']
POSITION_DIFFICULTY = {'1G_Flat': 1.0, '2G_Horizontal': 1.15, '3G_Vertical': 1.35,
                       '4G_Overhead': 1.55, '5G_Pipe': 1.7, '6G_Pipe45': 1.9}
PROCESS_DIFFICULTY = {'GMAW': 1.0, 'GTAW': 1.3, 'SMAW': 1.15}

# 6 core welding parameters that Skillveri tracks
PARAMETERS = [
    'work_angle',       # Torch angle relative to workpiece (degrees deviation from ideal)
    'travel_angle',     # Torch push/drag angle (degrees deviation)
    'travel_speed',     # Consistency of travel speed (score 0-100)
    'contact_tip_distance',  # Arc length / CTWD consistency (score 0-100)
    'arc_length',       # Arc stability score (0-100)
    'bead_quality',     # Overall bead appearance/consistency (0-100)
]

# Parameter correlations - some params are naturally correlated
# (good travel speed usually means good bead quality)
PARAM_CORRELATION_MATRIX = np.array([
    [1.00, 0.55, 0.30, 0.25, 0.20, 0.35],  # work_angle
    [0.55, 1.00, 0.35, 0.30, 0.25, 0.40],  # travel_angle
    [0.30, 0.35, 1.00, 0.45, 0.50, 0.65],  # travel_speed
    [0.25, 0.30, 0.45, 1.00, 0.60, 0.55],  # contact_tip_distance
    [0.20, 0.25, 0.50, 0.60, 1.00, 0.70],  # arc_length
    [0.35, 0.40, 0.65, 0.55, 0.70, 1.00],  # bead_quality
])

# Learner archetypes - models different learning patterns
LEARNER_ARCHETYPES = {
    'fast_learner':     {'learning_rate': (0.15, 0.04), 'ceiling': (88, 5), 'noise': (3, 1), 'weight': 0.15},
    'steady_learner':   {'learning_rate': (0.10, 0.03), 'ceiling': (82, 6), 'noise': (4, 1.5), 'weight': 0.35},
    'slow_starter':     {'learning_rate': (0.06, 0.02), 'ceiling': (78, 7), 'noise': (5, 2), 'weight': 0.25},
    'plateau_learner':  {'learning_rate': (0.12, 0.03), 'ceiling': (72, 5), 'noise': (6, 2), 'weight': 0.15},
    'struggling':       {'learning_rate': (0.04, 0.015), 'ceiling': (65, 8), 'noise': (8, 3), 'weight': 0.10},
}

INSTITUTIONS = [
    ('ITI_Chennai', 'India', 'Industrial'),
    ('ITI_Mumbai', 'India', 'Industrial'),
    ('NSDC_Delhi', 'India', 'Government'),
    ('Maruti_Training', 'India', 'Corporate'),
    ('Hyundai_Academy', 'India', 'Corporate'),
    ('Indian_Railways_WS', 'India', 'Corporate'),
    ('Tata_Motors_TC', 'India', 'Corporate'),
    ('CTE_Texas', 'USA', 'CTE_School'),
    ('CTE_Ohio', 'USA', 'CTE_School'),
    ('CTE_Michigan', 'USA', 'CTE_School'),
    ('Community_College_CA', 'USA', 'CTE_School'),
    ('Welding_Academy_UK', 'UK', 'Vocational'),
]


def logistic_learning_curve(session_num, learning_rate, ceiling, baseline=25):
    """
    Models skill acquisition using a logistic growth function.
    
    S(t) = baseline + (ceiling - baseline) / (1 + exp(-learning_rate * (t - inflection)))
    
    This is grounded in psychomotor learning theory:
    - Fitts & Posner (1967) three-stage model
    - Power law of practice (Newell & Rosenbloom, 1981)
    """
    inflection = 8 + np.random.normal(0, 2)  # Session where fastest learning happens
    score = baseline + (ceiling - baseline) / (1 + np.exp(-learning_rate * (session_num - inflection)))
    return score


def generate_correlated_params(base_scores, correlation_matrix):
    """
    Generate correlated parameter scores using Cholesky decomposition.
    This ensures realistic inter-parameter correlations.
    """
    n = len(base_scores)
    L = np.linalg.cholesky(correlation_matrix)
    uncorrelated = np.random.normal(0, 1, n)
    correlated_noise = L @ uncorrelated
    # Scale noise and add to base scores
    result = base_scores + correlated_noise * 5
    return np.clip(result, 0, 100)


def generate_learner_profile(learner_id):
    """Generate a complete learner profile with individual characteristics."""
    # Select archetype based on weights
    archetypes = list(LEARNER_ARCHETYPES.keys())
    weights = [LEARNER_ARCHETYPES[a]['weight'] for a in archetypes]
    archetype = np.random.choice(archetypes, p=weights)
    arch = LEARNER_ARCHETYPES[archetype]
    
    # Sample individual parameters from archetype distributions
    learning_rate = max(0.01, np.random.normal(*arch['learning_rate']))
    ceiling = min(98, max(50, np.random.normal(*arch['ceiling'])))
    noise_level = max(1, np.random.normal(*arch['noise']))
    
    # Per-parameter variation (some learners are better at angles vs speed)
    param_strengths = np.random.normal(1.0, 0.12, len(PARAMETERS))
    param_strengths = np.clip(param_strengths, 0.7, 1.3)
    
    # Select institution
    inst = INSTITUTIONS[np.random.randint(len(INSTITUTIONS))]
    
    # Demographics
    age = int(np.random.choice(
        [np.random.randint(16, 20), np.random.randint(20, 30), np.random.randint(30, 45)],
        p=[0.3, 0.5, 0.2]
    ))
    prior_experience = np.random.choice(
        ['none', 'basic_theory', 'some_manual', 'experienced'],
        p=[0.4, 0.25, 0.25, 0.1]
    )
    experience_bonus = {'none': 0, 'basic_theory': 3, 'some_manual': 8, 'experienced': 15}
    
    return {
        'learner_id': f'SKV-{learner_id:04d}',
        'archetype': archetype,
        'learning_rate': learning_rate,
        'ceiling': ceiling,
        'noise_level': noise_level,
        'param_strengths': param_strengths,
        'institution': inst[0],
        'country': inst[1],
        'institution_type': inst[2],
        'age': age,
        'prior_experience': prior_experience,
        'experience_bonus': experience_bonus[prior_experience],
        'primary_process': np.random.choice(WELDING_PROCESSES, p=[0.5, 0.3, 0.2]),
        'enrollment_date': datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 700)),
    }


def generate_sessions(profile, num_sessions=None):
    """Generate all training sessions for a learner."""
    if num_sessions is None:
        # Realistic session counts based on archetype
        base = {'fast_learner': 18, 'steady_learner': 25, 'slow_starter': 30,
                'plateau_learner': 28, 'struggling': 35}
        num_sessions = base[profile['archetype']] + np.random.randint(-5, 8)
        num_sessions = max(8, min(50, num_sessions))
    
    sessions = []
    current_date = profile['enrollment_date']
    
    # Track cumulative practice time
    total_practice_minutes = 0
    consecutive_days_practiced = 0
    last_practice_date = None
    dropout_risk = 0
    
    for session_idx in range(num_sessions):
        # Session scheduling (realistic gaps)
        if session_idx == 0:
            gap = 0
        else:
            # Mix of daily practice and gaps
            if np.random.random() < 0.6:
                gap = np.random.choice([1, 2], p=[0.7, 0.3])
            else:
                gap = np.random.choice([3, 4, 5, 7, 14], p=[0.3, 0.2, 0.2, 0.2, 0.1])
        
        current_date += timedelta(days=int(gap))
        
        # Track practice consistency
        if last_practice_date and (current_date - last_practice_date).days <= 2:
            consecutive_days_practiced += 1
        else:
            consecutive_days_practiced = max(0, consecutive_days_practiced - 1)
        last_practice_date = current_date
        
        # Session duration (minutes)
        session_duration = int(np.random.normal(45, 15))
        session_duration = max(15, min(90, session_duration))
        total_practice_minutes += session_duration
        
        # Number of weld exercises in this session
        num_exercises = int(np.random.normal(6, 2))
        num_exercises = max(2, min(12, num_exercises))
        
        # Select position and process for this session
        # Earlier sessions focus on easier positions
        if session_idx < 5:
            position_probs = [0.5, 0.3, 0.15, 0.05, 0.0, 0.0]
        elif session_idx < 15:
            position_probs = [0.2, 0.25, 0.25, 0.2, 0.07, 0.03]
        else:
            position_probs = [0.1, 0.15, 0.2, 0.25, 0.15, 0.15]
        
        position = np.random.choice(WELDING_POSITIONS, p=position_probs)
        process = profile['primary_process']
        
        # Calculate base score from learning curve
        effective_session = session_idx + profile['experience_bonus'] / 3
        base_score = logistic_learning_curve(
            effective_session,
            profile['learning_rate'],
            profile['ceiling'],
            baseline=25 + profile['experience_bonus']
        )
        
        # Apply difficulty modifiers
        pos_difficulty = POSITION_DIFFICULTY[position]
        proc_difficulty = PROCESS_DIFFICULTY[process]
        difficulty_penalty = (pos_difficulty * proc_difficulty - 1) * 15
        
        # Consistency bonus (regular practice helps)
        consistency_bonus = min(5, consecutive_days_practiced * 0.5)
        
        # Fatigue effect (later in long sessions)
        fatigue = max(0, (session_duration - 60) * 0.1)
        
        # Generate per-parameter scores with correlations
        param_base_scores = np.array([
            (base_score - difficulty_penalty + consistency_bonus - fatigue) * strength
            for strength in profile['param_strengths']
        ])
        param_base_scores = np.clip(param_base_scores, 5, 98)
        
        # Add correlated noise
        param_scores = generate_correlated_params(param_base_scores, PARAM_CORRELATION_MATRIX)
        
        # Add session-level noise (bad days, etc.)
        session_noise = np.random.normal(0, profile['noise_level'])
        param_scores += session_noise
        param_scores = np.clip(param_scores, 0, 100)
        
        # Overall session score (weighted average)
        weights = [0.15, 0.15, 0.20, 0.15, 0.15, 0.20]
        overall_score = np.average(param_scores, weights=weights)
        
        # Calculate improvement rate
        if session_idx > 0 and len(sessions) > 0:
            prev_score = sessions[-1]['overall_score']
            improvement_rate = overall_score - prev_score
        else:
            improvement_rate = 0
        
        session_data = {
            'learner_id': profile['learner_id'],
            'session_id': f'{profile["learner_id"]}-S{session_idx+1:03d}',
            'session_number': session_idx + 1,
            'session_date': current_date.strftime('%Y-%m-%d'),
            'institution': profile['institution'],
            'country': profile['country'],
            'institution_type': profile['institution_type'],
            'age': profile['age'],
            'prior_experience': profile['prior_experience'],
            'archetype': profile['archetype'],  # Ground truth for validation
            'welding_process': process,
            'welding_position': position,
            'position_difficulty': pos_difficulty,
            'session_duration_min': session_duration,
            'num_exercises': num_exercises,
            'total_practice_minutes': total_practice_minutes,
            'days_since_enrollment': (current_date - profile['enrollment_date']).days,
            'consecutive_practice_days': consecutive_days_practiced,
            'session_gap_days': int(gap),
        }
        
        # Add per-parameter scores
        for i, param in enumerate(PARAMETERS):
            session_data[f'score_{param}'] = round(param_scores[i], 2)
        
        session_data['overall_score'] = round(overall_score, 2)
        session_data['improvement_rate'] = round(improvement_rate, 2)
        
        sessions.append(session_data)
    
    return sessions


def determine_certification_outcome(sessions_df, learner_profile):
    """
    Determine if a learner would pass certification based on their trajectory.
    
    Certification criteria (modeled after AWS D1.1 and similar standards):
    - Must score >= 75 on at least 3 consecutive sessions
    - Must achieve >= 70 on all individual parameters in the final session
    - Must complete minimum 15 sessions
    - Overall trajectory must be improving (not declining)
    """
    if len(sessions_df) < 10:
        return 0, 0, 'insufficient_sessions'
    
    final_sessions = sessions_df.tail(5)
    avg_final_score = final_sessions['overall_score'].mean()
    
    # Check consecutive high scores
    scores = sessions_df['overall_score'].values
    max_consecutive_above_75 = 0
    current_streak = 0
    for s in scores:
        if s >= 75:
            current_streak += 1
            max_consecutive_above_75 = max(max_consecutive_above_75, current_streak)
        else:
            current_streak = 0
    
    # Check final session parameter minimums
    final_session = sessions_df.iloc[-1]
    param_scores = [final_session[f'score_{p}'] for p in PARAMETERS]
    all_params_above_70 = all(s >= 70 for s in param_scores)
    
    # Improvement trend (last 5 sessions)
    if len(sessions_df) >= 5:
        recent_trend = np.polyfit(range(5), final_sessions['overall_score'].values, 1)[0]
    else:
        recent_trend = 0
    
    # Calculate pass probability
    pass_prob = 0
    if avg_final_score >= 80 and max_consecutive_above_75 >= 3 and all_params_above_70:
        pass_prob = min(0.95, 0.7 + recent_trend * 0.05 + (avg_final_score - 80) * 0.01)
    elif avg_final_score >= 72 and max_consecutive_above_75 >= 2:
        pass_prob = min(0.7, 0.3 + (avg_final_score - 72) * 0.04)
    elif avg_final_score >= 65:
        pass_prob = min(0.35, 0.1 + (avg_final_score - 65) * 0.03)
    else:
        pass_prob = max(0.02, avg_final_score * 0.003)
    
    passed = 1 if np.random.random() < pass_prob else 0
    
    if passed:
        reason = 'passed'
    elif avg_final_score < 65:
        reason = 'low_overall_score'
    elif not all_params_above_70:
        weak_params = [p for p, s in zip(PARAMETERS, param_scores) if s < 70]
        reason = f'weak_params:{",".join(weak_params)}'
    elif max_consecutive_above_75 < 2:
        reason = 'inconsistent_performance'
    else:
        reason = 'marginal_fail'
    
    return passed, round(pass_prob, 3), reason


def generate_full_dataset(n_learners=600):
    """Generate the complete dataset."""
    print(f"Generating data for {n_learners} learners...")
    
    all_sessions = []
    learner_profiles = []
    certification_outcomes = []
    
    for i in range(n_learners):
        profile = generate_learner_profile(i + 1)
        sessions = generate_sessions(profile)
        all_sessions.extend(sessions)
        
        # Store profile
        profile_record = {k: v for k, v in profile.items() if k != 'param_strengths'}
        profile_record['param_strengths'] = profile['param_strengths'].tolist()
        profile_record['total_sessions'] = len(sessions)
        learner_profiles.append(profile_record)
        
        # Determine certification outcome
        sessions_df = pd.DataFrame(sessions)
        passed, pass_prob, reason = determine_certification_outcome(sessions_df, profile)
        certification_outcomes.append({
            'learner_id': profile['learner_id'],
            'certified': passed,
            'certification_probability': pass_prob,
            'outcome_reason': reason,
            'total_sessions': len(sessions),
            'final_overall_score': sessions_df['overall_score'].iloc[-1],
            'avg_last5_score': sessions_df['overall_score'].tail(5).mean(),
        })
    
    sessions_df = pd.DataFrame(all_sessions)
    profiles_df = pd.DataFrame(learner_profiles)
    outcomes_df = pd.DataFrame(certification_outcomes)
    
    print(f"Generated {len(sessions_df)} sessions for {n_learners} learners")
    print(f"Certification rate: {outcomes_df['certified'].mean():.1%}")
    print(f"Sessions per learner: {sessions_df.groupby('learner_id').size().describe()}")
    
    return sessions_df, profiles_df, outcomes_df


if __name__ == '__main__':
    sessions_df, profiles_df, outcomes_df = generate_full_dataset(600)
    
    # Save to CSV
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    sessions_df.to_csv(os.path.join(data_dir, 'training_sessions.csv'), index=False)
    profiles_df.to_csv(os.path.join(data_dir, 'learner_profiles.csv'), index=False)
    outcomes_df.to_csv(os.path.join(data_dir, 'certification_outcomes.csv'), index=False)
    
    print(f"\nData saved to {data_dir}/")
    print(f"  training_sessions.csv: {sessions_df.shape}")
    print(f"  learner_profiles.csv: {profiles_df.shape}")
    print(f"  certification_outcomes.csv: {outcomes_df.shape}")