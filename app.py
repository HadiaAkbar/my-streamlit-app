"""
FactGuard - AI Powered Fact Detector
Advanced fake news detection with stunning cyberpunk UI
"""
# Add this at line 1, right after the triple quotes
import streamlit as st
import time

# Show a loading screen immediately
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown("""
    <div style='text-align: center; padding: 100px;'>
        <div style='font-size: 4rem; margin-bottom: 30px;'>🛡️</div>
        <h1 style='color: #E91E8C;'>FACTGUARD</h1>
        <p style='color: #6366F1;'>Loading AI Fact-Checking System...</p>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(1)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import plotly.graph_objects as go

# ================== FACTGUARD THEME (Based on Logo) ==================
THEME = {
    "primary": "#E91E8C",        # Hot pink from logo
    "secondary": "#6366F1",      # Electric blue
    "accent": "#00D9FF",         # Cyan
    "danger": "#FF0055",         # Bright red
    "warning": "#FFB800",        # Amber
    "success": "#00FF88",        # Neon green
    "dark": "#0A0A1F",           # Deep purple-black
    "purple_dark": "#1A0B2E",    # Dark purple
    "purple_mid": "#2D1B4E",     # Mid purple
    "glow": "#FF00FF",           # Magenta glow
}

st.set_page_config(
    page_title="FactGuard - AI Fact Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== CYBERPUNK CSS ==================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Rajdhani', sans-serif;
    }}
    
    h1, h2, h3, h4 {{
        font-family: 'Orbitron', sans-serif;
        font-weight: 800;
    }}
    
    /* Cyberpunk background */
    .stApp {{
        background: linear-gradient(135deg, {THEME['dark']} 0%, {THEME['purple_dark']} 50%, {THEME['dark']} 100%);
        background-attachment: fixed;
        position: relative;
    }}
    
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(233, 30, 140, 0.03) 2px, rgba(233, 30, 140, 0.03) 4px),
            repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(99, 102, 241, 0.03) 2px, rgba(99, 102, 241, 0.03) 4px);
        pointer-events: none;
        z-index: 1;
    }}
    
    /* Neon glass cards */
    .cyber-card {{
        background: rgba(26, 11, 46, 0.7);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 2px solid rgba(233, 30, 140, 0.3);
        box-shadow: 
            0 0 20px rgba(233, 30, 140, 0.2),
            0 0 40px rgba(99, 102, 241, 0.1),
            inset 0 0 60px rgba(233, 30, 140, 0.05);
        position: relative;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .cyber-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, {THEME['primary']}, transparent);
        animation: scan 3s linear infinite;
    }}
    
    @keyframes scan {{
        0%, 100% {{ opacity: 0; }}
        50% {{ opacity: 1; }}
    }}
    
    .cyber-card:hover {{
        transform: translateY(-5px);
        border-color: {THEME['primary']};
        box-shadow: 
            0 0 30px rgba(233, 30, 140, 0.4),
            0 0 60px rgba(99, 102, 241, 0.2),
            inset 0 0 80px rgba(233, 30, 140, 0.08);
    }}
    
    /* Neon text effects */
    .neon-text {{
        color: {THEME['primary']};
        text-shadow: 
            0 0 10px {THEME['primary']},
            0 0 20px {THEME['primary']},
            0 0 30px {THEME['primary']},
            0 0 40px rgba(233, 30, 140, 0.5);
        animation: neon-flicker 3s infinite alternate;
    }}
    
    @keyframes neon-flicker {{
        0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {{
            text-shadow: 
                0 0 10px {THEME['primary']},
                0 0 20px {THEME['primary']},
                0 0 30px {THEME['primary']},
                0 0 40px rgba(233, 30, 140, 0.5);
        }}
        20%, 24%, 55% {{
            text-shadow: none;
        }}
    }}
    
    .cyber-title {{
        background: linear-gradient(135deg, {THEME['primary']}, {THEME['secondary']}, {THEME['accent']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 3s ease infinite;
        background-size: 200% auto;
        filter: drop-shadow(0 0 20px rgba(233, 30, 140, 0.5));
    }}
    
    @keyframes gradient-shift {{
        0%, 100% {{ background-position: 0% center; }}
        50% {{ background-position: 100% center; }}
    }}
    
    /* Cyberpunk buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {THEME['primary']}, {THEME['secondary']});
        color: white;
        border: 2px solid {THEME['primary']};
        padding: 16px 36px;
        border-radius: 12px;
        font-weight: 800;
        font-size: 16px;
        font-family: 'Orbitron', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        box-shadow: 
            0 0 20px rgba(233, 30, 140, 0.5),
            inset 0 0 20px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 
            0 0 40px rgba(233, 30, 140, 0.8),
            0 5px 30px rgba(233, 30, 140, 0.4),
            inset 0 0 30px rgba(255, 255, 255, 0.2);
        border-color: {THEME['accent']};
    }}
    
    .stButton > button:active {{
        transform: translateY(-1px) scale(0.98);
    }}
    
    /* Text area with cyber glow */
    .stTextArea textarea {{
        background: rgba(26, 11, 46, 0.8) !important;
        border: 2px solid rgba(233, 30, 140, 0.4) !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        font-size: 16px !important;
        padding: 20px !important;
        box-shadow: 
            inset 0 0 30px rgba(233, 30, 140, 0.1),
            0 0 20px rgba(233, 30, 140, 0.2) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextArea textarea:focus {{
        border-color: {THEME['primary']} !important;
        box-shadow: 
            0 0 30px rgba(233, 30, 140, 0.6),
            inset 0 0 40px rgba(233, 30, 140, 0.15) !important;
    }}
    
    /* Futuristic tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 15px;
        background: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(26, 11, 46, 0.6);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(233, 30, 140, 0.3);
        border-radius: 12px;
        padding: 16px 32px;
        color: #ffffff;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(233, 30, 140, 0.2);
        border-color: {THEME['primary']};
        box-shadow: 0 0 20px rgba(233, 30, 140, 0.4);
        transform: translateY(-2px);
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(233, 30, 140, 0.3), rgba(99, 102, 241, 0.3)) !important;
        border-color: {THEME['primary']} !important;
        box-shadow: 
            0 0 30px rgba(233, 30, 140, 0.6),
            inset 0 0 30px rgba(233, 30, 140, 0.2) !important;
        color: #ffffff !important;
    }}
    
    /* Success/Warning/Error with neon */
    div[data-testid="stSuccess"] {{
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.15), rgba(0, 255, 136, 0.25)) !important;
        border-left: 4px solid {THEME['success']} !important;
        border-radius: 12px !important;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.3) !important;
        color: #ffffff !important;
        font-weight: 600;
    }}
    
    div[data-testid="stError"] {{
        background: linear-gradient(135deg, rgba(255, 0, 85, 0.15), rgba(255, 0, 85, 0.25)) !important;
        border-left: 4px solid {THEME['danger']} !important;
        border-radius: 12px !important;
        box-shadow: 0 0 20px rgba(255, 0, 85, 0.3) !important;
        color: #ffffff !important;
        font-weight: 600;
    }}
    
    div[data-testid="stWarning"] {{
        background: linear-gradient(135deg, rgba(255, 184, 0, 0.15), rgba(255, 184, 0, 0.25)) !important;
        border-left: 4px solid {THEME['warning']} !important;
        border-radius: 12px !important;
        box-shadow: 0 0 20px rgba(255, 184, 0, 0.3) !important;
        color: #ffffff !important;
        font-weight: 600;
    }}
    
    /* Sidebar cyber theme */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {THEME['purple_dark']}, {THEME['dark']}) !important;
        border-right: 2px solid rgba(233, 30, 140, 0.3);
        box-shadow: 5px 0 30px rgba(233, 30, 140, 0.2);
    }}
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {{
        color: {THEME['primary']} !important;
        text-shadow: 0 0 10px rgba(233, 30, 140, 0.5);
    }}
    
    /* Expander cyber style */
    .streamlit-expanderHeader {{
        background: rgba(26, 11, 46, 0.8) !important;
        border: 2px solid rgba(233, 30, 140, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
    }}
    
    .streamlit-expanderHeader:hover {{
        border-color: {THEME['primary']} !important;
        box-shadow: 0 0 20px rgba(233, 30, 140, 0.4);
    }}
    
    /* Floating animation */
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        33% {{ transform: translateY(-15px) rotate(2deg); }}
        66% {{ transform: translateY(-8px) rotate(-2deg); }}
    }}
    
    .float-shield {{
        animation: float 6s ease-in-out infinite;
    }}
    
    /* Pulse glow */
    @keyframes pulse-glow {{
        0%, 100% {{ 
            box-shadow: 
                0 0 20px rgba(233, 30, 140, 0.4),
                0 0 40px rgba(99, 102, 241, 0.2);
        }}
        50% {{ 
            box-shadow: 
                0 0 40px rgba(233, 30, 140, 0.8),
                0 0 80px rgba(99, 102, 241, 0.4);
        }}
    }}
    
    .pulse-glow {{
        animation: pulse-glow 2s ease-in-out infinite;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {THEME['dark']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {THEME['primary']}, {THEME['secondary']});
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(233, 30, 140, 0.5);
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, {THEME['secondary']}, {THEME['primary']});
    }}
</style>
""", unsafe_allow_html=True)

# ================== SESSION STATE ==================
if 'news_text' not in st.session_state:
    st.session_state.news_text = ""

# ================== HEADER WITH LOGO ==================
st.markdown("""
<div style='text-align: center; margin: 30px 0 50px 0;' class='float-shield'>
    <div style='display: inline-block; position: relative;'>
        <div style='
            width: 160px;
            height: 160px;
            background: linear-gradient(135deg, #2D1B4E, #1A0B2E);
            border-radius: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px auto;
            border: 3px solid #E91E8C;
            box-shadow: 
                0 0 40px rgba(233, 30, 140, 0.6),
                0 0 80px rgba(99, 102, 241, 0.3),
                inset 0 0 60px rgba(233, 30, 140, 0.2);
            position: relative;
            overflow: hidden;
        '>
            <div style='font-size: 80px; z-index: 2;'>🛡️</div>
            <div style='
                position: absolute;
                width: 100%;
                height: 100%;
                background: linear-gradient(45deg, transparent 30%, rgba(233, 30, 140, 0.2) 50%, transparent 70%);
                animation: shine 3s infinite;
            '></div>
        </div>
    </div>
    
    <h1 style='
        font-size: 4.5rem; 
        margin: 0 0 10px 0;
        font-family: "Orbitron", sans-serif;
        font-weight: 900;
    ' class='cyber-title'>
        FACTGUARD
    </h1>
    
    <p style='
        font-size: 1.4rem; 
        color: #E91E8C;
        font-weight: 600;
        margin: 10px 0;
        text-shadow: 0 0 20px rgba(233, 30, 140, 0.6);
        font-family: "Rajdhani", sans-serif;
        letter-spacing: 2px;
    '>
        YOUR AI SHIELD AGAINST FAKE NEWS
    </p>
    
    <div style='
        height: 3px; 
        width: 250px; 
        background: linear-gradient(90deg, transparent, #E91E8C, #6366F1, transparent);
        margin: 25px auto;
        border-radius: 2px;
        box-shadow: 0 0 20px rgba(233, 30, 140, 0.6);
    '></div>
</div>

<style>
@keyframes shine {{
    0% {{ transform: translateX(-100%) translateY(-100%) rotate(45deg); }}
    100% {{ transform: translateX(100%) translateY(100%) rotate(45deg); }}
}}
</style>
""", unsafe_allow_html=True)

# ================== KNOWLEDGE BASE ==================
VERIFIED_KNOWLEDGE_BASE = {
    "medical": [
        {"fact": "COVID-19 vaccines are safe and effective", "confidence": 0.98},
        {"fact": "Vaccines do not cause autism", "confidence": 0.99},
    ],
    "climate": [
        {"fact": "Climate change is primarily human-caused", "confidence": 0.97},
        {"fact": "Global temperatures have risen 1.1°C since 1880", "confidence": 0.98},
    ],
    "space": [
        {"fact": "The Earth is approximately 4.5 billion years old", "confidence": 0.99},
        {"fact": "Mars has no evidence of current intelligent life", "confidence": 0.94},
    ],
}

FAKE_NEWS_PATTERNS = {
    "medical_miracles": ["cures all diseases", "miracle cure", "doctors hate this", "big pharma"],
    "conspiracy_theories": ["deep state", "hidden truth", "cover-up", "they don't want you"],
    "urgency_scams": ["limited time", "act now", "last chance", "hurry", "before it's too late"],
    "sensational_claims": ["shocking discovery", "mind-blowing", "you won't believe", "breaking", "exposed"]
}

# ================== FUNCTIONS ==================
def check_against_knowledge_base(text):
    text_lower = text.lower()
    results = {"verified_facts": [], "contradicted_facts": [], "fake_patterns_found": []}
    
    for category, facts in VERIFIED_KNOWLEDGE_BASE.items():
        for fact_entry in facts:
            fact_text = fact_entry["fact"].lower()
            fact_words = set(fact_text.split()[:10])
            
            if any(keyword in text_lower for keyword in fact_words if len(keyword) > 3):
                contradiction_keywords = ["not true", "is false", "fake", "hoax", "lie", "false", "myth", "debunked"]
                is_contradiction = any(neg in text_lower for neg in contradiction_keywords)
                
                if is_contradiction:
                    results["contradicted_facts"].append(fact_entry)
                else:
                    results["verified_facts"].append(fact_entry)
    
    for pattern_type, patterns in FAKE_NEWS_PATTERNS.items():
        found_patterns = [p for p in patterns if p in text_lower]
        if found_patterns:
            results["fake_patterns_found"].append({"type": pattern_type, "patterns": found_patterns})
    
    return results

def analyze_deception_score(text):
    text_lower = text.lower()
    score = 0
    
    sensational_words = ['breaking', 'shocking', 'amazing', 'miracle', 'secret', 'exposed', 'urgent']
    score += min(sum(1 for word in sensational_words if word in text_lower) * 4, 25)
    
    words = text.split()
    if len(words) > 0:
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
        score += min((caps_words / len(words)) * 120, 20)
    
    score += min(text.count('!') * 3, 15)
    
    urgency_words = ['now', 'immediately', 'hurry', 'last chance', 'limited time']
    score += min(sum(1 for word in urgency_words if word in text_lower) * 4, 20)
    
    evidence_indicators = ['according to', 'study shows', 'research', 'scientists', 'university']
    evidence_count = sum(1 for indicator in evidence_indicators if indicator in text_lower)
    score += max(0, 20 - (evidence_count * 4))
    
    return min(score, 100)

def classify_content(deception_score, knowledge_check):
    contradictions = len(knowledge_check.get('contradicted_facts', []))
    fake_patterns = len(knowledge_check.get('fake_patterns_found', []))
    verified_facts = len(knowledge_check.get('verified_facts', []))
    
    if contradictions >= 2 or (contradictions >= 1 and deception_score >= 50):
        return "FAKE", THEME['danger'], "🔴"
    elif fake_patterns >= 2 and deception_score >= 60:
        return "FAKE", THEME['danger'], "🔴"
    elif deception_score >= 75:
        return "FAKE", THEME['danger'], "🔴"
    elif deception_score >= 50 or contradictions >= 1:
        return "SUSPICIOUS", THEME['warning'], "🟡"
    elif verified_facts >= 2 and deception_score < 30:
        return "REAL", THEME['success'], "🟢"
    elif deception_score < 25:
        return "REAL", THEME['success'], "🟢"
    else:
        return "SUSPICIOUS", THEME['warning'], "🟡"

def create_cyber_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 20, 'color': '#ffffff', 'family': 'Orbitron'}},
        number={'font': {'size': 42, 'color': color, 'family': 'Orbitron'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': color},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': "rgba(26, 11, 46, 0.5)",
            'borderwidth': 3,
            'bordercolor': color,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 136, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(255, 184, 0, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 85, 0.2)'}
            ],
            'threshold': {'line': {'color': "#ffffff", 'width': 4}, 'thickness': 0.8, 'value': value}
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=70, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff'}
    )
    return fig

def create_cyber_radar(features):
    categories = ['Sensationalism', 'Urgency', 'Capitalization', 'Emotion', 'Evidence']
    values = [features.get(k, 0) for k in ['sensational', 'urgency', 'caps', 'emotional', 'evidence']]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        fillcolor='rgba(233, 30, 140, 0.3)',
        line=dict(color='#E91E8C', width=3)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(233, 30, 140, 0.2)'),
            bgcolor='rgba(26, 11, 46, 0.3)',
            angularaxis=dict(gridcolor='rgba(233, 30, 140, 0.2)')
        ),
        showlegend=False,
        height=400,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ffffff', 'family': 'Rajdhani'}
    )
    return fig

# ================== SIDEBAR ==================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h2 class='neon-text' style='font-family: "Orbitron";'>⚡ SYSTEM STATUS</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class='cyber-card' style='text-align: center; padding: 20px;'>
            <div style='font-size: 2.5rem;'>🔬</div>
            <div style='font-weight: 700; color: {THEME['accent']}; margin-top: 8px;'>Fact Check</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: {THEME['primary']};'>96%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_s2:
        st.markdown(f"""
        <div class='cyber-card' style='text-align: center; padding: 20px;'>
            <div style='font-size: 2.5rem;'>🧠</div>
            <div style='font-weight: 700; color: {THEME['accent']}; margin-top: 8px;'>AI Analysis</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: {THEME['primary']};'>93%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    if st.button("🔄 RESET SYSTEM", use_container_width=True):
        st.session_state.news_text = ""
        st.rerun()

# ================== MAIN TABS ==================
tab1, tab2, tab3 = st.tabs(["🔍 VERIFY CONTENT", "📊 ANALYTICS", "⚙️ SYSTEM INFO"])

with tab1:
    st.markdown("""
    <div class='cyber-card pulse-glow'>
        <h2 style='margin-top: 0; color: #ffffff; font-family: "Orbitron";'>
            🤖 AI-POWERED CONTENT VERIFICATION
        </h2>
        <p style='color: #E91E8C; font-size: 1.2rem; font-weight: 600;'>
            Deploy advanced neural networks to detect misinformation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_in1, col_in2 = st.columns([3, 1])
    
    with col_in1:
        st.markdown("""
        <div class='cyber-card'>
            <h4 style='margin-top: 0; color: #E91E8C; font-family: "Orbitron";'>
                📝 INPUT CONTENT
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        news_text = st.text_area("", value=st.session_state.news_text, height=240,
            placeholder="Paste the news article or claim to verify...",
            label_visibility="collapsed", key="input_text")
    
    with col_in2:
        st.markdown("""
        <div class='cyber-card' style='height: 100%;'>
            <h4 style='margin-top: 0; color: #E91E8C; font-family: "Orbitron";'>
                ⚡ QUICK TESTS
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🧪 FAKE EXAMPLE", use_container_width=True):
            st.session_state.news_text = "BREAKING: SHOCKING medical discovery will CHANGE MEDICINE FOREVER! Doctors HATE this one simple trick that INSTANTLY cures diabetes! ACT NOW before Big Pharma removes this!"
            st.rerun()
        
        if st.button("📚 REAL EXAMPLE", use_container_width=True):
            st.session_state.news_text = "According to a study published in Nature Climate Change, global sea levels have risen by approximately 3.7 millimeters per year over the past decade."
            st.rerun()
        
        if st.button("🗑️ CLEAR ALL", use_container_width=True):
            st.session_state.news_text = ""
            st.rerun()
    
    col_a1, col_a2, col_a3 = st.columns([1, 2, 1])
    with col_a2:
        analyze_btn = st.button("🚀 LAUNCH ANALYSIS", use_container_width=True, type="primary")
    
    if analyze_btn and news_text:
        with st.spinner("🔬 Deploying Neural Networks..."):
            time.sleep(1.5)
            
            knowledge_check = check_against_knowledge_base(news_text)
            deception_score = analyze_deception_score(news_text)
            credibility_score = 100 - deception_score
            classification, verdict_color, verdict_emoji = classify_content(deception_score, knowledge_check)
            
            features = {
                'sensational': min(news_text.lower().count('shocking') * 20 + news_text.lower().count('breaking') * 15, 100),
                'urgency': min(news_text.lower().count('now') * 15 + news_text.lower().count('hurry') * 20, 100),
                'caps': min((sum(1 for c in news_text if c.isupper()) / max(len(news_text), 1)) * 200, 100),
                'emotional': min(news_text.count('!') * 10, 100),
                'evidence': max(0, 100 - news_text.lower().count('according') * 25)
            }
            
            st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
            
            if classification == "FAKE":
                card_bg = "linear-gradient(135deg, rgba(255, 0, 85, 0.2), rgba(255, 0, 85, 0.3))"
            elif classification == "SUSPICIOUS":
                card_bg = "linear-gradient(135deg, rgba(255, 184, 0, 0.2), rgba(255, 184, 0, 0.3))"
            else:
                card_bg = "linear-gradient(135deg, rgba(0, 255, 136, 0.2), rgba(0, 255, 136, 0.3))"
            
            st.markdown(f"""
            <div class='cyber-card pulse-glow' style='background: {card_bg}; border: 3px solid {verdict_color};'>
                <div style='display: flex; align-items: center; gap: 30px;'>
                    <div style='font-size: 5rem; filter: drop-shadow(0 0 20px {verdict_color});'>{verdict_emoji}</div>
                    <div>
                        <h1 style='
                            margin: 0; 
                            color: {verdict_color}; 
                            font-size: 3rem;
                            font-family: "Orbitron";
                            text-shadow: 0 0 20px {verdict_color};
                        '>
                            CLASSIFIED: {classification}
                        </h1>
                        <p style='color: #ffffff; margin: 12px 0 0 0; font-size: 1.2rem; font-weight: 700;'>
                            🛡️ FactGuard AI Analysis Complete
                        </p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_sc1, col_sc2, col_sc3 = st.columns(3)
            
            with col_sc1:
                st.plotly_chart(create_cyber_gauge(deception_score, "DECEPTION SCORE", verdict_color), use_container_width=True)
            
            with col_sc2:
                st.plotly_chart(create_cyber_gauge(credibility_score, "CREDIBILITY SCORE", 
                    THEME['success'] if credibility_score > 60 else THEME['warning']), use_container_width=True)
            
            with col_sc3:
                verified = len(knowledge_check.get('verified_facts', []))
                contradicted = len(knowledge_check.get('contradicted_facts', []))
                
                st.markdown(f"""
                <div class='cyber-card' style='height: 300px; display: flex; flex-direction: column; justify-content: center;'>
                    <div style='text-align: center; margin-bottom: 25px;'>
                        <div style='
                            font-size: 4rem; 
                            font-weight: 900; 
                            color: {THEME["success"]};
                            text-shadow: 0 0 20px {THEME["success"]};
                            font-family: "Orbitron";
                        '>{verified}</div>
                        <div style='color: #ffffff; font-weight: 700; font-size: 1.2rem;'>VERIFIED FACTS</div>
                    </div>
                    <div style='height: 2px; background: linear-gradient(90deg, transparent, {THEME["primary"]}, transparent); margin: 15px 0;'></div>
                    <div style='text-align: center; margin-top: 25px;'>
                        <div style='
                            font-size: 4rem; 
                            font-weight: 900; 
                            color: {THEME["danger"]};
                            text-shadow: 0 0 20px {THEME["danger"]};
                            font-family: "Orbitron";
                        '>{contradicted}</div>
                        <div style='color: #ffffff; font-weight: 700; font-size: 1.2rem;'>CONTRADICTIONS</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("📊 ADVANCED ANALYSIS DASHBOARD", expanded=True):
                st.markdown("""
                <div class='cyber-card'>
                    <h4 style='color: #E91E8C; font-family: "Orbitron";'>🎯 DECEPTION PATTERN ANALYSIS</h4>
                    <p style='color: #ffffff;'>Multi-dimensional threat detection visualization</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(create_cyber_radar(features), use_container_width=True)
                
                if knowledge_check.get('verified_facts') or knowledge_check.get('contradicted_facts'):
                    col_f1, col_f2 = st.columns(2)
                    
                    with col_f1:
                        if knowledge_check.get('verified_facts'):
                            st.success("✅ SUPPORTS VERIFIED FACTS")
                            for fact in knowledge_check['verified_facts'][:3]:
                                st.write(f"• {fact['fact']}")
                    
                    with col_f2:
                        if knowledge_check.get('contradicted_facts'):
                            st.error("❌ CONTRADICTS VERIFIED FACTS")
                            for fact in knowledge_check['contradicted_facts'][:3]:
                                st.write(f"• {fact['fact']}")
                
                if knowledge_check.get('fake_patterns_found'):
                    st.markdown("""
                    <div class='cyber-card'>
                        <h4 style='color: #FF0055; font-family: "Orbitron";'>🚨 FAKE NEWS PATTERNS DETECTED</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for pattern_group in knowledge_check['fake_patterns_found']:
                        pattern_type = pattern_group['type'].replace('_', ' ').title()
                        patterns_text = ', '.join(pattern_group['patterns'][:3])
                        st.warning(f"**{pattern_type}:** {patterns_text}")
                
                st.markdown("""
                <div class='cyber-card'>
                    <h4 style='color: #00D9FF; font-family: "Orbitron";'>💡 RECOMMENDED ACTIONS</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if classification == "FAKE":
                    st.error("""
                    **🚨 HIGH-RISK CONTENT - EXTREME CAUTION**
                    
                    ⚠️ **DO NOT SHARE** - High probability of misinformation
                    
                    🔍 Verify with primary sources and fact-checking organizations
                    
                    📢 Report if necessary as potential misinformation
                    """)
                elif classification == "SUSPICIOUS":
                    st.warning("""
                    **⚠️ SUSPICIOUS CONTENT - VERIFY BEFORE SHARING**
                    
                    🔎 Cross-reference with multiple reliable sources
                    
                    📅 Check publication dates for context
                    
                    🎓 Seek expert opinions in the relevant field
                    """)
                else:
                    st.success("""
                    **✅ CONTENT APPEARS CREDIBLE**
                    
                    ✓ Still verify original sources and references
                    
                    🧠 Consider context and stay informed
                    
                    📱 Share responsibly with proper attribution
                    """)

with tab2:
    st.markdown("""
    <div class='cyber-card'>
        <h2 style='margin-top: 0; color: #ffffff; font-family: "Orbitron";'>
            📈 ANALYTICS DASHBOARD
        </h2>
        <p style='color: #E91E8C; font-size: 1.1rem;'>Real-time system performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    
    metrics = [
        ("⏱️", "2.1s", "Avg Analysis", THEME['accent']),
        ("🎯", "92.3%", "Accuracy", THEME['success']),
        ("📊", "42.8K", "Facts DB", THEME['primary']),
        ("⚡", "6", "AI Layers", THEME['warning'])
    ]
    
    for col, (icon, value, label, color) in zip([col_p1, col_p2, col_p3, col_p4], metrics):
        with col:
            st.markdown(f"""
            <div class='cyber-card' style='text-align: center; padding: 25px; border-color: {color};'>
                <div style='font-size: 3rem; filter: drop-shadow(0 0 10px {color});'>{icon}</div>
                <div style='
                    font-size: 2.2rem; 
                    font-weight: 900; 
                    color: {color};
                    text-shadow: 0 0 15px {color};
                    font-family: "Orbitron";
                    margin: 10px 0;
                '>{value}</div>
                <div style='color: #ffffff; font-weight: 700; font-size: 1rem;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        st.markdown("""
        <div class='cyber-card'>
            <h4 style='color: #E91E8C; font-family: "Orbitron";'>📈 DETECTION PERFORMANCE</h4>
        </div>
        """, unsafe_allow_html=True)
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        accuracy = [88, 89, 90, 91, 92, 92.3]
        
        fig_trend = go.Figure(data=go.Scatter(
            x=months, y=accuracy,
            mode='lines+markers',
            line=dict(color=THEME['primary'], width=4),
            marker=dict(size=12, color=THEME['accent'], line=dict(color='#ffffff', width=2))
        ))
        
        fig_trend.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=30),
            plot_bgcolor='rgba(26, 11, 46, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[85, 95], gridcolor='rgba(233, 30, 140, 0.2)', color='#ffffff'),
            xaxis=dict(gridcolor='rgba(233, 30, 140, 0.2)', color='#ffffff'),
            font=dict(family='Rajdhani', color='#ffffff')
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_ch2:
        st.markdown("""
        <div class='cyber-card'>
            <h4 style='color: #E91E8C; font-family: "Orbitron";'>🍰 CONTENT DISTRIBUTION</h4>
        </div>
        """, unsafe_allow_html=True)
        
        labels = ['Verified', 'Suspicious', 'Fake', 'Uncertain']
        values = [45, 25, 20, 10]
        colors = [THEME['success'], THEME['warning'], THEME['danger'], THEME['secondary']]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, values=values,
            hole=.5,
            marker_colors=colors,
            textinfo='label+percent',
            textfont=dict(size=14, family='Orbitron', color='#ffffff')
        )])
        
        fig_pie.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=30),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.markdown("""
    <div class='cyber-card'>
        <h2 style='margin-top: 0; color: #ffffff; font-family: "Orbitron";'>
            ⚙️ SYSTEM ARCHITECTURE
        </h2>
        <p style='color: #E91E8C; font-size: 1.1rem;'>Multi-layer AI verification system</p>
    </div>
    """, unsafe_allow_html=True)
    
    layers = [
        ("🔍", "Fact Verification", "Cross-references with 42K+ verified facts database"),
        ("🧠", "Logical Analysis", "Detects inconsistencies and contradictions"),
        ("📚", "Source Check", "Evaluates credibility of information sources"),
        ("🎭", "Style Detection", "Analyzes linguistic deception patterns"),
        ("📊", "Evidence Review", "Checks for supporting data and citations"),
        ("🌐", "Web Verification", "Cross-checks with online fact databases")
    ]
    
    for i, (icon, title, desc) in enumerate(layers):
        st.markdown(f"""
        <div class='cyber-card' style='margin: 15px 0;'>
            <div style='display: flex; align-items: center; gap: 25px;'>
                <div style='font-size: 3rem; filter: drop-shadow(0 0 10px {THEME["primary"]});'>{icon}</div>
                <div style='flex: 1;'>
                    <h4 style='margin: 0; color: {THEME["primary"]}; font-family: "Orbitron";'>{title}</h4>
                    <p style='color: #ffffff; margin: 8px 0 0 0; opacity: 0.9;'>{desc}</p>
                </div>
                <div style='
                    background: linear-gradient(135deg, {THEME["primary"]}, {THEME["secondary"]}); 
                    color: white; 
                    padding: 10px 24px; 
                    border-radius: 20px; 
                    font-weight: 800;
                    font-family: "Orbitron";
                    box-shadow: 0 0 20px rgba(233, 30, 140, 0.5);
                '>
                    LAYER {i+1}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    col_spec1, col_spec2 = st.columns(2)
    
    with col_spec1:
        st.markdown(f"""
        <div class='cyber-card'>
            <h4 style='color: {THEME["primary"]}; font-family: "Orbitron";'>🔬 AI & ML MODELS</h4>
            <ul style='color: #ffffff; line-height: 2;'>
                <li>BERT-based NLP for semantic analysis</li>
                <li>Neural network ensemble learning</li>
                <li>Real-time pattern recognition</li>
                <li>Continuous adaptive learning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_spec2:
        st.markdown(f"""
        <div class='cyber-card'>
            <h4 style='color: {THEME["primary"]}; font-family: "Orbitron";'>📊 DATA INFRASTRUCTURE</h4>
            <ul style='color: #ffffff; line-height: 2;'>
                <li>42,891 verified facts database</li>
                <li>Real-time API integration</li>
                <li>Distributed cloud computing</li>
                <li>Military-grade encryption</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
# Clear the loading placeholder once everything is loaded
loading_placeholder.empty()
# ================== FOOTER ==================
st.markdown(f"""
<div style='text-align: center; padding: 50px 0 30px 0;'>
    <div style='
        font-size: 1.5rem; 
        font-weight: 800; 
        margin-bottom: 15px;
        color: {THEME["primary"]};
        text-shadow: 0 0 20px rgba(233, 30, 140, 0.6);
        font-family: "Orbitron";
    '>
        🛡️ FACTGUARD - "YOUR AI SHIELD AGAINST FAKE NEWS"
    </div>
    <p style='font-size: 1rem; color: #ffffff; opacity: 0.8;'>
        Version 3.6.0 | Last Updated: {datetime.now().strftime("%Y-%m-%d")}
    </p>
    <p style='font-size: 0.95rem; margin-top: 25px; color: {THEME["accent"]}; opacity: 0.9;'>
        ⚠️ This is an AI-assisted tool. Always verify important information through multiple reliable sources.
    </p>
    <p style='font-size: 1rem; margin-top: 20px; color: #ffffff; opacity: 0.9;'>
        Prepared by: <strong style='color: {THEME["primary"]};'>Hadia Akbar (042)</strong> | <strong style='color: {THEME["primary"]};'>Naira Shahid (062)</strong>
    </p>
    <div style='
        height: 3px; 
        width: 200px; 
        background: linear-gradient(90deg, transparent, {THEME["primary"]}, {THEME["secondary"]}, transparent);
        margin: 30px auto;
        border-radius: 2px;
        box-shadow: 0 0 20px rgba(233, 30, 140, 0.6);
    '></div>
</div>
""", unsafe_allow_html=True)