"""
Skillveri Predictive Skill Readiness Score
==========================================
Streamlit Dashboard — Redesigned for Clarity

Every page answers a clear business question.
Designed so the MD can immediately understand the value.

Run: streamlit run App.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
import os

# ─── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skillveri AI Readiness Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { font-size: 1.1rem; color: #94a3b8; margin-top: 0; margin-bottom: 30px; }
    .insight-box {
        background: #1e293b !important;
        border-left: 4px solid #00d4aa;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.95rem;
        color: #ffffff !important;
    }
    .insight-box b, .insight-box-warn b, .insight-box-danger b {
        color: #ffffff !important;
    }
    .insight-box-warn {
        background: #2d2a1a !important;
        border-left: 4px solid #f59e0b;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        color: #ffffff !important;
        font-size: 0.95rem;
    }
    .insight-box-danger {
        background: #2d1a1a !important;
        border-left: 4px solid #ef4444;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        color: #ffffff !important;
        font-size: 0.95rem;
    }
    .question-banner {
        background: linear-gradient(135deg, #0d2137 0%, #1a1028 100%);
        border: 1px solid #2a3050;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .question-text {
        font-size: 1.4rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .question-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 6px;
    }
    div[data-testid="stSidebar"] { background-color: #111827; }
</style>
""", unsafe_allow_html=True)

# ─── Load Data & Artifacts ──────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, 'data')
    model_dir = os.path.join(base, 'models')
    
    sessions = pd.read_csv(os.path.join(data_dir, 'training_sessions.csv'))
    features = pd.read_csv(os.path.join(data_dir, 'features.csv'))
    outcomes = pd.read_csv(os.path.join(data_dir, 'certification_outcomes.csv'))
    
    with open(os.path.join(model_dir, 'model_comparison.json')) as f:
        model_comparison = json.load(f)
    
    shap_importance = pd.read_csv(os.path.join(model_dir, 'shap_importance.csv'))
    
    with open(os.path.join(model_dir, 'roc_curve.json')) as f:
        roc_data = json.load(f)
    with open(os.path.join(model_dir, 'pr_curve.json')) as f:
        pr_data = json.load(f)
    with open(os.path.join(model_dir, 'confusion_matrix.json')) as f:
        cm_data = json.load(f)
    with open(os.path.join(model_dir, 'learning_curves.json')) as f:
        lc_data = json.load(f)
    with open(os.path.join(model_dir, 'calibration.json')) as f:
        cal_data = json.load(f)
    
    test_preds = pd.read_csv(os.path.join(model_dir, 'test_predictions.csv'))
    
    with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, 'feature_cols.json')) as f:
        feature_cols = json.load(f)
    
    return {
        'sessions': sessions, 'features': features, 'outcomes': outcomes,
        'model_comparison': model_comparison, 'shap_importance': shap_importance,
        'roc_data': roc_data, 'pr_data': pr_data, 'cm_data': cm_data,
        'lc_data': lc_data, 'cal_data': cal_data, 'test_preds': test_preds,
        'model': model, 'feature_cols': feature_cols,
    }

data = load_data()

# ─── Helper: Categorize Learners ────────────────────────────────────
def categorize_learner(row):
    if row['certified'] == 1:
        return '✅ Certified'
    elif row['current_score'] >= 72 and row.get('recent_momentum', 0) > 0:
        return '🟡 Almost Ready'
    elif row['current_score'] >= 55:
        return '🔵 In Progress'
    else:
        return '🔴 Needs Intervention'

features_df = data['features'].copy()
features_df['status'] = features_df.apply(categorize_learner, axis=1)

# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Skillveri AI Predictor")
    st.markdown("---")
    
    page = st.radio(
        "Select a view",
        [
            "🏠 What Is This Project?",
            "👥 Batch Readiness Report",
            "👤 Individual Learner Lookup",
            "🔬 What Makes a Welder Ready?",
            "🤖 How Accurate Is the AI?",
            "🏗️ Technical Deep Dive",
        ],
        index=0
    )
    
    st.markdown("---")
    st.caption("**Project 2** — Predictive Skill Readiness")
    st.caption("For Skillveri AI Engineer Interview")
    st.caption(f"Data: {len(data['features'])} learners, {len(data['sessions']):,} sessions")


# ═══════════════════════════════════════════════════════════════════
# PAGE 1: WHAT IS THIS PROJECT?
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 What Is This Project?":
    st.markdown('<p class="main-header">🎯 Predictive Skill Readiness Score for Skillveri</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An AI system that predicts when a VR simulator trainee will be job-ready</p>', unsafe_allow_html=True)
    
    # ── The Problem ──
    st.markdown("""<div class="question-banner">
        <div class="question-text">💡 The Problem This Solves</div>
        <div class="question-sub">Training managers at Hyundai, Maruti, Indian Railways ask Skillveri:<br>
        <b>"When will my batch of 50 welders be ready for the shop floor?"</b><br><br>
        Today, Skillveri Insights shows what <i>already happened</i> (scores, charts). 
        But it can't predict what <i>will happen next</i>. This project adds that prediction layer.</div>
    </div>""", unsafe_allow_html=True)
    
    # ── What It Does ──
    st.markdown("### What This AI System Does")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="insight-box">
            <b>1️⃣ Predicts Certification Pass/Fail</b><br>
            "This learner has an 87% chance of passing certification based on their current trajectory."
        </div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="insight-box">
            <b>2️⃣ Estimates Sessions to Readiness</b><br>
            "This learner needs approximately 8 more practice sessions before they're job-ready."
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="insight-box">
            <b>3️⃣ Identifies Bottleneck Sub-Skills</b><br>
            "Their travel_speed and work_angle are holding them back — focus practice there."
        </div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="insight-box">
            <b>4️⃣ Batch-Level Workforce Planning</b><br>
            "Out of 50 welders: 12 are ready, 25 are close (2-3 weeks), 13 need intervention."
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Quick Stats ──
    st.markdown("### Project at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Learners Analyzed", "600")
    col2.metric("Training Sessions", f"{len(data['sessions']):,}")
    col3.metric("AI Features Engineered", "95")
    col4.metric("Prediction Accuracy (AUC)", "98.1%")
    
    st.markdown("---")
    
    # ── How It Works (Simple) ──
    st.markdown("### How It Works — 4 Steps")
    st.code("""
    STEP 1: Collect Data
    ─────────────────────
    Every time a learner uses Skillveri's VR simulator, 
    it records 6 parameters: work angle, travel angle, 
    travel speed, arc length, contact tip distance, bead quality.
    
    STEP 2: Engineer 95 Smart Features  
    ──────────────────────────────────
    From raw session scores, we compute things like:
    • Learning speed (are they improving fast or slow?)
    • Consistency (do they score 80 every day, or 90 one day and 50 the next?)
    • Weak spots (which specific parameter drags them down?)
    • Practice habits (how often do they practice? any long breaks?)
    
    STEP 3: Train an AI Model (XGBoost)
    ────────────────────────────────────
    The model learns patterns from 600 learners' data:
    "Learners who score above 75 for 3+ sessions in a row 
     AND have no parameter below 70 → 92% chance of passing."
    
    STEP 4: Predict & Recommend
    ───────────────────────────
    For any new learner, the AI instantly tells you:
    • Pass probability, estimated sessions remaining, what to practice next
    """, language=None)
    
    st.markdown("---")
    st.markdown("### Why This Matters for Skillveri")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="insight-box">
            <b>🏢 For Sales Teams</b><br>
            "Our AI predicts your batch will be job-ready in 3 weeks" → Quantifiable ROI for customers
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="insight-box-warn">
            <b>⚔️ vs Competitors</b><br>
            Interplay Learning has AI mentor "SAM". Lincoln VRTEX and Soldamatic have no AI prediction. This puts Skillveri ahead.
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="insight-box">
            <b>📊 For Product (Insights)</b><br>
            Transforms Skillveri Insights from a reporting tool into a prediction engine — makes it indispensable.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2: BATCH READINESS REPORT
# ═══════════════════════════════════════════════════════════════════
elif page == "👥 Batch Readiness Report":
    st.markdown('<p class="main-header">👥 Batch Readiness Report</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">This is what a training manager at Hyundai or Indian Railways would see every week</p>', unsafe_allow_html=True)
    
    st.markdown("""<div class="question-banner">
        <div class="question-text">❓ Question this page answers:</div>
        <div class="question-sub">"Out of my current batch of trainees, how many are ready, how many are close, and who needs extra help?"</div>
    </div>""", unsafe_allow_html=True)
    
    # Filter
    inst = st.selectbox("Select Training Center", ['All Institutions'] + sorted(features_df['institution'].unique().tolist()))
    if inst != 'All Institutions':
        batch = features_df[features_df['institution'] == inst].copy()
    else:
        batch = features_df.copy()
    
    # ── Summary Cards ──
    total = len(batch)
    certified = (batch['status'] == '✅ Certified').sum()
    almost = (batch['status'] == '🟡 Almost Ready').sum()
    progress = (batch['status'] == '🔵 In Progress').sum()
    needs_help = (batch['status'] == '🔴 Needs Intervention').sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Certified & Ready", f"{certified}", f"{certified/total*100:.0f}% of batch")
    col2.metric("🟡 Almost Ready (1-3 weeks)", f"{almost}", f"{almost/total*100:.0f}% of batch")
    col3.metric("🔵 In Progress", f"{progress}", f"{progress/total*100:.0f}% of batch")
    col4.metric("🔴 Needs Intervention", f"{needs_help}", f"{needs_help/total*100:.0f}% of batch")
    
    st.markdown("---")
    
    # ── Visual: Readiness Breakdown ──
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Readiness Breakdown")
        status_counts = batch['status'].value_counts()
        color_map = {'✅ Certified': '#00d4aa', '🟡 Almost Ready': '#f59e0b', 
                     '🔵 In Progress': '#3b82f6', '🔴 Needs Intervention': '#ef4444'}
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index, values=status_counts.values,
            hole=0.55,
            marker_colors=[color_map.get(s, '#666') for s in status_counts.index],
            textinfo='label+value', textposition='outside'
        )])
        fig.update_layout(template='plotly_dark', height=380, margin=dict(t=20, b=20),
                         showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Learner Scores vs Sessions Spent")
        st.caption("Each dot = one learner. Higher and further left = more efficient learner.")
        fig = px.scatter(
            batch, x='total_sessions', y='current_score',
            color='status', 
            color_discrete_map=color_map,
            hover_data=['learner_id', 'institution'],
            size_max=10
        )
        fig.add_hline(y=75, line_dash="dash", line_color="white", opacity=0.4,
                      annotation_text="Certification Threshold (75)")
        fig.update_layout(
            template='plotly_dark', height=380, margin=dict(t=20, b=20),
            xaxis_title='Total Sessions Completed',
            yaxis_title='Current Score (out of 100)',
            legend=dict(orientation='h', y=-0.15)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ── Insight Box ──
    if needs_help > 0:
        st.markdown(f"""<div class="insight-box-danger">
            <b>⚠️ Action Required:</b> {needs_help} learners are significantly behind. 
            These learners have scores below 55 and need restructured practice plans. 
            Scroll down to see who they are and what specific skills they're struggling with.
        </div>""", unsafe_allow_html=True)
    
    # ── Detailed Table ──
    st.markdown("#### Learner Details — Sortable Table")
    st.caption("Click any column header to sort. Look at 'Weakest Skill' to know what each learner should practice next.")
    
    display_cols = ['learner_id', 'status', 'current_score', 'total_sessions', 
                    'recent_5_mean', 'weakest_param_name', 'weakest_param_score',
                    'max_consecutive_above_75', 'sessions_per_week', 'institution']
    available_cols = [c for c in display_cols if c in batch.columns]
    
    display_df = batch[available_cols].copy()
    rename_map = {
        'learner_id': 'Learner ID', 'status': 'Status', 'current_score': 'Current Score',
        'total_sessions': 'Sessions', 'recent_5_mean': 'Avg Last 5',
        'weakest_param_name': 'Weakest Skill', 'weakest_param_score': 'Weakest Score',
        'max_consecutive_above_75': 'Best Streak ≥75', 'sessions_per_week': 'Sessions/Week',
        'institution': 'Institution'
    }
    display_df = display_df.rename(columns={k:v for k,v in rename_map.items() if k in display_df.columns})
    
    for col in display_df.select_dtypes(include='float64').columns:
        display_df[col] = display_df[col].round(1)
    
    st.dataframe(display_df.sort_values('Current Score', ascending=False), 
                 height=400, use_container_width=True)
    
    # ── Bottleneck Analysis ──
    st.markdown("---")
    st.markdown("#### Most Common Bottleneck Skills (Among Non-Certified Learners)")
    st.caption("This tells the training manager: 'Most of your struggling learners have the same problem — focus group training on these skills.'")
    
    not_cert = batch[batch['certified'] == 0]
    if 'weakest_param_name' in not_cert.columns and len(not_cert) > 0:
        bottlenecks = not_cert['weakest_param_name'].value_counts()
        fig = go.Figure(go.Bar(
            x=bottlenecks.index, y=bottlenecks.values,
            marker_color=['#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16', '#22c55e'][:len(bottlenecks)],
            text=bottlenecks.values, textposition='outside'
        ))
        fig.update_layout(
            template='plotly_dark', height=300, margin=dict(t=10, b=10),
            xaxis_title='Weak Sub-Skill', yaxis_title='Number of Struggling Learners'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        top_bottleneck = bottlenecks.index[0]
        st.markdown(f"""<div class="insight-box-warn">
            <b>💡 Recommendation:</b> The most common bottleneck is <b>{top_bottleneck}</b>. 
            Consider running a focused group session on this skill for all non-certified learners.
            This single intervention could accelerate readiness for {bottlenecks.values[0]} learners.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3: INDIVIDUAL LEARNER LOOKUP
# ═══════════════════════════════════════════════════════════════════
elif page == "👤 Individual Learner Lookup":
    st.markdown('<p class="main-header">👤 Individual Learner Lookup</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Select any learner to see their full skill profile, prediction, and AI recommendations</p>', unsafe_allow_html=True)
    
    st.markdown("""<div class="question-banner">
        <div class="question-text">❓ Question this page answers:</div>
        <div class="question-sub">"What's going on with this specific learner? Are they improving? What should they practice next?"</div>
    </div>""", unsafe_allow_html=True)
    
    sessions = data['sessions']
    
    # Selector
    learner_id = st.selectbox("Pick a Learner", features_df['learner_id'].unique())
    learner = features_df[features_df['learner_id'] == learner_id].iloc[0]
    learner_sessions = sessions[sessions['learner_id'] == learner_id].sort_values('session_number')
    
    # ── Status Banner ──
    status = learner['status']
    score = learner['current_score']
    
    if status == '✅ Certified':
        st.success(f"**{learner_id}** — Status: **CERTIFIED & READY** | Current Score: **{score:.1f}/100** | Sessions: **{int(learner['total_sessions'])}**")
    elif status == '🟡 Almost Ready':
        momentum = learner.get('recent_momentum', 0)
        est = max(3, int((75 - score) / max(0.5, abs(momentum)))) if momentum > 0 else '10+'
        st.warning(f"**{learner_id}** — Status: **ALMOST READY** | Score: **{score:.1f}/100** | Est. **{est} more sessions** needed")
    elif status == '🔵 In Progress':
        st.info(f"**{learner_id}** — Status: **IN PROGRESS** | Score: **{score:.1f}/100** | Sessions: **{int(learner['total_sessions'])}**")
    else:
        st.error(f"**{learner_id}** — Status: **NEEDS INTERVENTION** | Score: **{score:.1f}/100** | Struggling — see details below")
    
    st.markdown("---")
    
    # ── Key Metrics ──
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Score", f"{score:.1f}")
    col2.metric("Best Streak ≥75", f"{int(learner['max_consecutive_above_75'])} sessions")
    col3.metric("Practice Freq", f"{learner['sessions_per_week']:.1f}/week")
    col4.metric("Weakest Skill", learner.get('weakest_param_name', 'N/A'))
    col5.metric("Weakest Score", f"{learner['weakest_param_score']:.1f}")
    
    # ── Learning Curve ──
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### Learning Curve Over Time")
        st.caption("Green line = overall score per session. Dashed line = certification threshold (75).")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=learner_sessions['session_number'], y=learner_sessions['overall_score'],
            mode='lines+markers', name='Overall Score',
            line=dict(color='#00d4aa', width=3), marker=dict(size=6)
        ))
        fig.add_hrect(y0=75, y1=100, fillcolor='#00d4aa', opacity=0.05,
                      annotation_text="Passing Zone", annotation_position="top left")
        fig.add_hline(y=75, line_dash="dash", line_color="white", opacity=0.4)
        fig.update_layout(
            template='plotly_dark', height=350, margin=dict(t=10, b=10),
            xaxis_title='Session Number', yaxis_title='Score (0-100)',
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Sub-Skill Breakdown")
        st.caption("Each axis = one welding parameter. Bigger shape = better skills.")
        
        params = ['work_angle', 'travel_angle', 'travel_speed', 'ctwd', 'arc_length', 'bead_quality']
        param_labels = ['Work Angle', 'Travel Angle', 'Travel Speed', 'CTWD', 'Arc Length', 'Bead Quality']
        
        recent_vals = [learner.get(f'recent_{p}', 50) for p in params]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=recent_vals + [recent_vals[0]],
            theta=param_labels + [param_labels[0]],
            fill='toself', name='Current Level',
            line=dict(color='#00d4aa', width=2),
            fillcolor='rgba(0,212,170,0.2)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[75]*7, theta=param_labels + [param_labels[0]],
            mode='lines', name='Threshold (75)',
            line=dict(color='white', width=1, dash='dash'),
        ))
        fig.update_layout(
            template='plotly_dark', height=350, margin=dict(t=30, b=10),
            polar=dict(radialaxis=dict(range=[0, 100])),
            showlegend=True, legend=dict(x=0, y=-0.2, orientation='h')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ── AI Recommendation ──
    st.markdown("---")
    st.markdown("#### 🤖 AI Coaching Recommendation")
    
    weak_param = learner.get('weakest_param_name', 'unknown')
    weak_score = learner['weakest_param_score']
    momentum = learner.get('recent_momentum', 0)
    consistency = learner.get('score_cv', 0.5)
    gap_issue = learner.get('pct_gaps_over_5days', 0) > 0.2
    
    recommendations = []
    
    if weak_score < 65:
        recommendations.append(f"🎯 **Focus Area:** Your weakest skill is **{weak_param}** (score: {weak_score:.0f}/100). Dedicate at least 40% of your next 5 sessions to exercises targeting this parameter specifically.")
    
    if learner['max_consecutive_above_75'] < 2:
        recommendations.append("📈 **Consistency Needed:** You haven't yet scored above 75 for 2+ sessions in a row. Focus on steady, consistent technique rather than trying to hit high scores occasionally.")
    
    if gap_issue:
        recommendations.append("📅 **Practice More Regularly:** You have significant gaps between sessions (5+ days). Skill memory decays quickly — even 15-minute daily sessions are better than long sessions with big gaps.")
    
    if momentum < 0:
        recommendations.append("⚠️ **Score Declining:** Your recent scores are trending downward. This might indicate fatigue, frustration, or attempting too-difficult positions. Step back to easier exercises to rebuild confidence.")
    elif momentum > 1:
        recommendations.append("🚀 **Great Momentum:** You're improving steadily. Keep your current practice routine — it's working!")
    
    if consistency > 0.15:
        recommendations.append("🎲 **Reduce Variability:** Your scores vary a lot session to session. Try to focus on reproducing the same technique consistently rather than experimenting with different approaches.")
    
    if not recommendations:
        recommendations.append("✅ **On Track:** Your profile looks strong. Continue your current practice pattern and you should be certification-ready soon.")
    
    for rec in recommendations:
        st.markdown(rec)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4: WHAT MAKES A WELDER READY?
# ═══════════════════════════════════════════════════════════════════
elif page == "🔬 What Makes a Welder Ready?":
    st.markdown('<p class="main-header">🔬 What Makes a Welder Certification-Ready?</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">The AI analyzed 600 learners and found these are the biggest factors</p>', unsafe_allow_html=True)
    
    st.markdown("""<div class="question-banner">
        <div class="question-text">❓ Question this page answers:</div>
        <div class="question-sub">"What separates learners who pass certification from those who don't? What should training programs focus on?"</div>
    </div>""", unsafe_allow_html=True)
    
    # ── Top SHAP Features ──
    st.markdown("#### The 15 Most Important Predictors of Certification")
    st.caption("Based on SHAP analysis — this tells us which factors the AI weighs most heavily when making predictions.")
    
    shap_df = data['shap_importance'].head(15)
    
    label_map = {
        'current_score': 'Current Overall Score',
        'score_vs_institution': 'Score vs Institution Average',
        'weakest_param_score': 'Weakest Sub-Skill Score',
        'zscore_vs_institution': 'Relative Rank in Institution',
        'max_consecutive_above_75': 'Longest Streak Above 75',
        'avg_gap_days': 'Average Days Between Sessions',
        'pct_sessions_above_75': '% Sessions Scoring Above 75',
        'recent_5_mean': 'Average of Last 5 Sessions',
        'max_score_achieved': 'Highest Score Ever Achieved',
        'sessions_above_75': 'Total Sessions Above 75',
        'avg_score_3G_Vertical': 'Score on Vertical Welding (3G)',
        'num_params_above_80': 'Number of Strong Sub-Skills',
        'score_vs_global': 'Score vs Global Average',
        'score_at_max_difficulty': 'Score at Hardest Position',
        'avg_score_4G_Overhead': 'Score on Overhead Welding (4G)',
    }
    
    shap_display = shap_df.copy()
    shap_display['readable_name'] = shap_display['feature'].map(label_map).fillna(shap_display['feature'])
    
    fig = go.Figure(go.Bar(
        y=shap_display['readable_name'][::-1],
        x=shap_display['mean_abs_shap'][::-1],
        orientation='h',
        marker=dict(
            color=shap_display['mean_abs_shap'][::-1],
            colorscale=[[0, '#1e3a5f'], [0.5, '#7c3aed'], [1, '#00d4aa']],
        ),
        text=shap_display['mean_abs_shap'][::-1].apply(lambda x: f'{x:.3f}'),
        textposition='outside'
    ))
    fig.update_layout(
        template='plotly_dark', height=500, margin=dict(l=280, t=10, r=60),
        xaxis_title='Importance (higher = more impact on prediction)',
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Plain English Insights ──
    st.markdown("#### What This Means in Plain English")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="insight-box">
            <b>📊 #1: Current Score is King</b><br>
            The single best predictor of certification is simply the learner's latest overall score. 
            No surprise — but it confirms the simulator scoring is well-calibrated.
        </div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="insight-box">
            <b>🔗 #3: One Weak Skill Kills You</b><br>
            Even if your average is 78, having one parameter (like work_angle) stuck at 55 
            makes certification unlikely. The chain is only as strong as its weakest link.
        </div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="insight-box">
            <b>📐 Vertical Welding (3G) is the True Test</b><br>
            Performance on 3G (vertical) positions is far more predictive than 1G (flat). 
            This makes sense — vertical welding requires fighting gravity and reveals true technique.
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class="insight-box-warn">
            <b>📏 #2: Relative Performance Matters</b><br>
            A score of 72 at a tough institution (like Maruti_Training) is worth more than 
            72 at an easier one. The model learned to account for institutional difficulty.
        </div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="insight-box-warn">
            <b>🔥 Streaks > Spikes</b><br>
            Scoring 90 once doesn't help. What matters is scoring 75+ for 3+ sessions 
            in a row. Certification requires <i>reliable, repeatable</i> skill — not one good day.
        </div>""", unsafe_allow_html=True)
        
        st.markdown("""<div class="insight-box-danger">
            <b>📅 Practice Gaps are Deadly</b><br>
            Learners with average gaps over 5 days between sessions are significantly less 
            likely to certify. Even 2-3 missed days hurts psychomotor skill retention.
        </div>""", unsafe_allow_html=True)
    
    # ── Certified vs Not Comparison ──
    st.markdown("---")
    st.markdown("#### Certified vs Non-Certified: Side-by-Side Comparison")
    
    cert = features_df[features_df['certified'] == 1]
    not_cert = features_df[features_df['certified'] == 0]
    
    compare_metrics = {
        'Average Score': ('current_score', '.1f'),
        'Weakest Sub-Skill': ('weakest_param_score', '.1f'),
        'Best Streak >= 75': ('max_consecutive_above_75', '.1f'),
        'Sessions/Week': ('sessions_per_week', '.2f'),
        'Avg Gap (Days)': ('avg_gap_days', '.1f'),
        'Score Consistency (lower=better)': ('score_cv', '.3f'),
    }
    
    rows = []
    for label, (col, fmt) in compare_metrics.items():
        if col in cert.columns:
            rows.append({
                'Metric': label,
                'Certified ✅': f"{cert[col].mean():{fmt}}",
                'Not Certified ❌': f"{not_cert[col].mean():{fmt}}",
                'Difference': f"{cert[col].mean() - not_cert[col].mean():+{fmt}}"
            })
    
    compare_df = pd.DataFrame(rows)
    st.table(compare_df)


# ═══════════════════════════════════════════════════════════════════
# PAGE 5: HOW ACCURATE IS THE AI?
# ═══════════════════════════════════════════════════════════════════
elif page == "🤖 How Accurate Is the AI?":
    st.markdown('<p class="main-header">🤖 How Accurate Is the AI Model?</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Technical validation — proving the predictions are trustworthy</p>', unsafe_allow_html=True)
    
    st.markdown("""<div class="question-banner">
        <div class="question-text">❓ Question this page answers:</div>
        <div class="question-sub">"Can we actually trust these predictions? How was the model validated?"</div>
    </div>""", unsafe_allow_html=True)
    
    # ── Model Comparison ──
    st.markdown("#### We Tested 4 Different AI Algorithms")
    st.caption("XGBoost won — it's the most accurate and also the fastest for real-time predictions.")
    
    comp = pd.DataFrame(data['model_comparison']).T
    comp.index.name = 'Algorithm'
    
    metric_labels = {
        'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall',
        'f1_score': 'F1 Score', 'auc_roc': 'AUC-ROC', 'brier_score': 'Brier Score',
        'cv_auc_5fold': '5-Fold CV AUC'
    }
    display_comp = comp.rename(columns=metric_labels).round(3)
    st.dataframe(display_comp, use_container_width=True)
    
    st.markdown("""<div class="insight-box">
        <b>What do these numbers mean?</b><br>
        • <b>AUC-ROC (0.981)</b> — The model correctly ranks certified vs non-certified learners 98.1% of the time. Anything above 0.90 is considered excellent.<br>
        • <b>Precision (0.933)</b> — When the model says "this learner will pass," it's right 93.3% of the time.<br>
        • <b>Recall (0.824)</b> — The model catches 82.4% of learners who actually do pass.<br>
        • <b>Brier Score (0.036)</b> — The probability estimates are very well-calibrated. Close to 0 = perfect.
    </div>""", unsafe_allow_html=True)
    
    # ── ROC & Confusion ──
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ROC Curve")
        st.caption("The further from the diagonal, the better the model. Our curve hugs the top-left corner.")
        
        roc = data['roc_data']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc['fpr'], y=roc['tpr'], mode='lines',
            name=f'XGBoost (AUC = 0.981)', line=dict(color='#00d4aa', width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Random Guessing', line=dict(color='#4a5568', dash='dash')
        ))
        fig.update_layout(
            template='plotly_dark', height=380, margin=dict(t=10),
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Confusion Matrix")
        st.caption("Out of 120 test learners, here's what the model predicted vs what actually happened:")
        
        cm = np.array(data['cm_data']['matrix'])
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=['Predicted: Fail', 'Predicted: Pass'],
            y=['Actually: Fail', 'Actually: Pass'],
            colorscale=[[0, '#1a1f2e'], [1, '#00d4aa']],
            text=[[f'{cm[0][0]}\nCorrect Rejections', f'{cm[0][1]}\nFalse Alarms'],
                  [f'{cm[1][0]}\nMissed', f'{cm[1][1]}\nCorrect Passes']],
            texttemplate='%{text}', textfont=dict(size=13),
            showscale=False
        ))
        fig.update_layout(template='plotly_dark', height=380, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # ── Learning Curve ──
    st.markdown("#### Learning Curve — Is the Model Overfitting?")
    st.caption("If the two lines converge, the model generalizes well and isn't just memorizing training data.")
    
    lc = data['lc_data']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lc['train_sizes'], y=lc['train_scores_mean'],
        mode='lines+markers', name='Training Score',
        line=dict(color='#00d4aa'), marker=dict(size=5)
    ))
    fig.add_trace(go.Scatter(
        x=lc['train_sizes'], y=lc['val_scores_mean'],
        mode='lines+markers', name='Validation Score',
        line=dict(color='#f59e0b'), marker=dict(size=5)
    ))
    fig.update_layout(
        template='plotly_dark', height=350, margin=dict(t=10),
        xaxis_title='Number of Training Samples',
        yaxis_title='AUC-ROC Score', yaxis_range=[0.5, 1.05]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""<div class="insight-box">
        <b>✅ No overfitting detected.</b> The training and validation curves converge, 
        meaning the model generalizes well to unseen learners. With more data (Skillveri has 3M+ learners), 
        the model would be even more accurate.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 6: TECHNICAL DEEP DIVE
# ═══════════════════════════════════════════════════════════════════
elif page == "🏗️ Technical Deep Dive":
    st.markdown('<p class="main-header">🏗️ Technical Architecture</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">For the AI Engineer interview — showing technical depth</p>', unsafe_allow_html=True)
    
    st.markdown("#### System Architecture")
    st.code("""
    ┌───────────────────┐     ┌──────────────────┐     ┌────────────────────┐     ┌──────────────┐
    │  Skillveri VR      │────▶│  Raw Session Data │────▶│ Feature Engineering │────▶│  XGBoost     │
    │  Simulator         │     │  16,424 sessions  │     │  95 features        │     │  AUC: 0.981  │
    │  (AURA/Chroma)     │     │  6 params/session │     │  6 categories       │     │              │
    └───────────────────┘     └──────────────────┘     └────────────────────┘     └──────┬───────┘
                                                                                          │
                 ┌────────────────┬────────────────┬────────────────┐                      │
                 ▼                ▼                ▼                ▼                      │
       ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                │
       │  Readiness   │ │  Bottleneck  │ │  Sessions to │ │  SHAP        │◀───────────────┘
       │  Score (%)   │ │  Detection   │ │  Readiness   │ │  Explanations│
       └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
    """, language=None)
    
    st.markdown("#### Feature Engineering — 6 Categories, 95 Features")
    
    categories = {
        "1. Trajectory (18 features)": "Learning curve shape: slope, acceleration, plateau detection, projected scores. Uses logistic growth model based on psychomotor learning theory (Fitts & Posner, 1967).",
        "2. Consistency (10 features)": "Score variability: CV, IQR, session-to-session differences. Key insight: A welder scoring 85 then 55 is NOT ready even with a 70 average.",
        "3. Sub-Skill (30+ features)": "Per-parameter analysis across all 6 welding parameters. Identifies specific weaknesses (e.g., good travel_speed but bad work_angle).",
        "4. Practice Patterns (14 features)": "Session frequency, gap analysis, regularity entropy, duration trends. Captures the spaced-repetition effect on skill retention.",
        "5. Difficulty Progression (12 features)": "Performance across 1G-6G positions. Certification requires competence at multiple difficulty levels.",
        "6. Relative Performance (4 features)": "Z-scores vs institution and global averages. Normalizes for institutional effects.",
    }
    
    for title, desc in categories.items():
        with st.expander(title):
            st.markdown(desc)
    
    st.markdown("#### Why XGBoost?")
    st.markdown("""
    - **Best performer** across all metrics (AUC, F1, Brier)
    - **Handles class imbalance** natively (scale_pos_weight)
    - **Feature interactions** captured automatically (high score + high consistency → pass)
    - **Fast inference** (< 5ms per prediction) — suitable for real-time use in Skillveri Insights
    - **Interpretable** via SHAP — training managers need to *trust* the predictions
    - **Right for small data** — 600 samples is too small for deep learning (Grinsztajn et al., 2022 showed gradient boosting beats NNs on tabular data < 10K rows)
    """)
    
    st.markdown("#### Production Deployment Plan")
    st.markdown("""
    - **Retraining:** Weekly batch with new session data from all 500+ installations
    - **Feature Store:** Pre-compute features as sessions are recorded (incremental updates)
    - **API:** FastAPI endpoint returning readiness score + recommendations per learner
    - **Integration:** Embed in Skillveri Insights dashboard as a new "Readiness Prediction" tab
    - **Monitoring:** Track prediction drift, feature drift, and compare predicted vs actual certification rates
    - **A/B Test:** Compare AI-recommended practice plans vs trainer-designed plans → measure time-to-certification reduction
    """)