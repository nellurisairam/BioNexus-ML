# app_streamlit.py
# Streamlit dashboard for Bioreactor ML Prediction using saved sklearn Pipeline artifacts
# Works with artifacts produced by predict.py (model .joblib + feature_schema.json)

import json
import math
from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load, dump
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import streamlit.components.v1 as components
import base64
from database_utils import (
    get_authenticator_config, update_user_approval, add_user, init_db, list_users, 
    delete_user, update_user_role, save_prediction, get_user_history, delete_history_item,
    get_alert_config, save_alert_config, send_email_alert
)
from threading import Thread
import bcrypt

# 1.0 ROOT DIR SETUP
ROOT_DIR = Path(__file__).parent

# 1. MUST BE FIRST
st.set_page_config(page_title="Bioreactor ML Dashboard", layout="wide")

# 1.1 IMAGE HELPER
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_img_with_href(local_img_path):
    img_format = local_img_path.split(".")[-1]
    bin_str = get_base64_of_bin_file(local_img_path)
    return f"data:image/{img_format};base64,{bin_str}"

# 1.5 CUSTOM CSS DESIGN
def set_design(theme="Light", is_authenticated=False):
    bg_img = ""
    try:
        # Check assets folder for background
        bg_path = ROOT_DIR / "assets/background.png"
        if bg_path.exists():
            bg_data = get_img_with_href(str(bg_path))
            bg_img = f"url({bg_data})"
    except:
        pass

    # Theme configuration
    if theme == "Dark":
        bg_overlay = "rgba(16, 42, 67, 0.85)"
        text_color = "#f0f4f8"
        card_bg = "rgba(255, 255, 255, 0.1)"
        card_border = "rgba(255, 255, 255, 0.15)"
        sidebar_bg = "rgba(0, 0, 0, 0.4)"
        metric_val = "#00d1ff"
        input_bg = "rgba(16, 42, 67, 0.95)"
        input_border = card_border
        base_bg_color = "#0a1929" # Dark background base for overrides
    else:
        bg_overlay = "rgba(240, 244, 248, 0.8)"
        text_color = "#102a43"
        card_bg = "rgba(255, 255, 255, 0.7)"
        card_border = "rgba(255, 255, 255, 0.5)"
        sidebar_bg = "rgba(255, 255, 255, 0.3)"
        metric_val = "#004a8c"
        input_bg = "#ffffff"
        input_border = "rgba(16, 42, 67, 0.2)" # Darker border for visibility
        base_bg_color = "#f0f4f8" # Light background base

    # If not authenticated, we MUST ensure text is light as background is dark
    if not is_authenticated:
        bg_overlay = "rgba(10, 25, 41, 0.8)" # Darker slate background for login
        text_color = "#ffffff" # Force white text for visibility
        card_bg = "rgba(255, 255, 255, 0.1)"
        card_border = "rgba(255, 255, 255, 0.2)"
        input_bg = "rgba(10, 25, 41, 0.95)"
        input_border = "rgba(255, 255, 255, 0.2)"
        sidebar_bg = "rgba(10, 25, 41, 0.8)" # Ensure sidebar is dark enough for white text
        base_bg_color = "#0a1929"

    st.markdown(f"""
    <style>
    :root {{
        --text-color: {text_color} !important;
        --background-color: {base_bg_color} !important;
        --secondary-background-color: {sidebar_bg} !important;
        --primary-color: {metric_val} !important;
    }}
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, .stApp, .stMarkdown, p, li {{
        font-family: 'Inter', sans-serif;
        color: {text_color} !important;
    }}

    /* Widget Labels & Captions */
    .stWidget label, .stCaption, .stSlider label {{
        color: {text_color} !important;
        font-weight: 600 !important;
    }}

    .stApp {{
        background: {bg_overlay} {bg_img};
        background-size: cover;
        background-blend-mode: overlay;
        background-attachment: fixed;
    }}
    
    [data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid {card_border} !important;
    }}

    .stExpander, div[data-testid="stMetric"], .stTabs, div[data-testid="stDataFrame"], .stAlert {{
        background-color: {card_bg} !important;
        border-radius: 12px !important;
        border: 1px solid {card_border} !important;
        padding: 15px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }}

    .stExpander:hover, div[data-testid="stMetric"]:hover {{
        border-color: {metric_val} !important;
        box-shadow: 0 0 20px {metric_val}22 !important;
    }}

    /* Expander explicit header transparency to kill white backgrounds */
    .stExpander details, .stExpander summary, [data-testid="stExpander"] details, [data-testid="stExpander"] summary {{
        background-color: transparent !important;
    }}
    .stExpander summary p, .stExpander summary span, .stExpander summary svg {{
        color: {text_color} !important;
        fill: {text_color} !important;
    }}

    div[data-testid="stMetricValue"] {{
        font-weight: 700 !important;
        color: {metric_val} !important;
        text-shadow: 0 0 10px {metric_val}33;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }}

    /* Premium Alert Box Styling */
    div.stAlert {{
        border: none !important;
        border-left: 5px solid {metric_val} !important;
        background-color: {card_bg} !important;
    }}
    div.stAlert[data-baseweb="alert"] {{
        background-color: {card_bg} !important;
    }}

    /* Global Tab Overrides */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px !important;
        background-color: transparent !important;
        padding: 10px 0 !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px !important;
        background-color: {card_bg} !important;
        border-radius: 10px 10px 0 0 !important;
        border: 1px solid {card_border} !important;
        border-bottom: none !important;
        padding: 0 25px !important;
        color: {text_color} !important;
        transition: all 0.2s ease !important;
        opacity: 0.6;
    }}

    .stTabs [aria-selected="true"] {{
        opacity: 1 !important;
        background-color: {card_bg} !important;
        border-bottom: 3px solid {metric_val} !important;
        transform: translateY(-2px);
    }}
    
    /* Ensure Dataframe and Alert text is readable in both modes */
    [data-testid="stTable"] td, [data-testid="stTable"] th, div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th, .stExpander p, .stAlert p, .stAlert h3, .stAlert li {{
        color: {text_color} !important;
    }}

    /* 1. Base input wrapper - giving it the unified background */
    div[data-baseweb="base-input"], div[data-baseweb="select"] {{
        background-color: {input_bg} !important;
        border-radius: 8px !important;
        border: 1px solid {input_border} !important;
    }}

    /* 2. Strip backgrounds from all children so they blend into the wrapper */
    div[data-baseweb="base-input"] *, div[data-baseweb="select"] * {{
        background-color: transparent !important;
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
    }}

    /* 3. Strip the nasty border that Streamlit puts on the actual input tag */
    .stTextInput input, .stNumberInput input, textarea, select, input {{
        border: none !important;
        box-shadow: none !important;
    }}

    /* Specific fix for Dropdown (Selectbox) items and popovers */
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], li[role="option"] {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
    }}
    
    li[role="option"]:hover {{
        background-color: {metric_val} !important;
        color: #ffffff !important;
    }}
    
    /* Ensure icons match text */
    div[data-baseweb="base-input"] svg, div[data-baseweb="select"] svg {{
        fill: {text_color} !important;
    }}

    /* Placeholders */
    ::placeholder {{
        color: {text_color} !important;
        opacity: 0.6 !important;
        -webkit-text-fill-color: {text_color} !important;
    }}

    /* Universal Widget & Component Text Visibility (Sliders, Uploaders, Metrics, Toggles) */
    .stCheckbox label, .stToggle label, .stRadio label, .stWidgetLabel p, label, .stSlider p, div[data-baseweb="slider"] div, .stMetric label {{
        color: {text_color} !important;
        font-weight: 600 !important;
    }}
    
    .stFileUploader label, .stFileUploader section, .stFileUploader section p, .stFileUploader section small {{
        color: {text_color} !important;
    }}

    /* Tooltips, Glossary Help Text, and Markdown Code elements */
    .stTooltipContent, div[data-testid="stTooltipContent"], .stHelp, div[data-testid="stHelp"] *, code {{
        color: {text_color} !important;
    }}
    code {{
        background-color: rgba(0, 0, 0, 0.2) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }}

    /* Ensure ALL buttons and their text are always visible with high contrast */
    .stButton > button, .stButton > button p, .stButton > button span, .stForm submit button, .stForm submit button p {{
        color: white !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .stButton > button {{
        border-radius: 10px !important;
        border: none !important;
        background: linear-gradient(135deg, #00d1ff 0%, #0054A0 100%) !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 15px rgba(0, 209, 255, 0.3) !important;
    }}
    
    /* Full-width buttons for specific forms */
    .stForm button, .stButton.full-width button {{
        width: 100% !important;
    }}
    
    .stButton > button:hover {{
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(0, 209, 255, 0.4) !important;
        background: linear-gradient(135deg, #00e5ff 0%, #0066cc 100%) !important;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{background: transparent !important;}}
    
    .block-container {{
        padding-top: 2rem !important;
    }}

    /* Specific Login Card styling for clarity on bg */
    [data-testid="stForm"], .stForm {{
        background: {card_bg} !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 12px !important;
        border: 1px solid {card_border} !important;
        padding: 30px !important;
        box-shadow: 0 20px 50px rgba(0,0,0,0.2) !important;
    }}
    [data-testid="stForm"] label, .stForm label {{
        color: {text_color} !important;
    }}
    /* Force buttons in auth forms to stay blue/white */
    .stForm button, .stForm button p {{
        background: linear-gradient(135deg, #0054A0 0%, #003a6d 100%) !important;
        color: white !important;
    }}

    /* Vertical alignment helper for management cards */
    .user-card {{
        padding: 10px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        margin-bottom: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }}
    
    /* Optimize buttons in cards */
    .user-card button {{
        margin-top: 5px !important;
    }}

    /* ── Responsive: Tablet (max 1024px) ── */
    @media (max-width: 1024px) {{
        .block-container {{
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }}
        h1 {{ font-size: 1.8rem !important; }}
        h2 {{ font-size: 1.4rem !important; }}
    }}

    /* ── Responsive: Mobile (max 768px) ── */
    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 1rem !important;
        }}
        /* Stack metrics vertically */
        div[data-testid="stMetric"] {{
            min-width: 100% !important;
        }}
        /* Make tabs horizontally scrollable instead of wrapping */
        .stTabs [data-baseweb="tab-list"] {{
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            gap: 6px !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            min-width: fit-content !important;
            padding: 0 12px !important;
            height: 40px !important;
            font-size: 0.8rem !important;
        }}
        /* Login branding scaling */
        h1 {{ font-size: 2.5rem !important; }}
        h2 {{ font-size: 1.2rem !important; }}
        /* Forms full-width */
        [data-testid="stForm"] {{
            padding: 16px !important;
        }}
        /* Expanded sidebar doesn't overlap content */
        .stButton > button {{
            padding: 8px 14px !important;
            font-size: 0.85rem !important;
        }}
    }}

    /* ── Responsive: Small Mobile (max 480px) ── */
    @media (max-width: 480px) {{
        .block-container {{
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }}
        h1 {{ font-size: 2rem !important; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.4rem !important; }}
    }}
    </style>
    """, unsafe_allow_html=True)

# Call design early for login page
set_design(is_authenticated=st.session_state.get("authentication_status", False))

# 2. AUTHENTICATION SETUP
init_db()
config = get_authenticator_config()

# Final fail-safe: Ensure admin user is injected into the config if missing
if 'admin' not in config['credentials']['usernames']:
    config['credentials']['usernames']['admin'] = {
        'username': 'admin',
        'email': 'admin@example.com',
        'name': 'System Admin',
        'password': bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode(),
        'roles': ['admin', 'user'],
        'role': 'admin',
        'approved': True,
        'logged_in': False
    }

authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days']
)

# --- BRANDING ON LOGIN PAGE ---
if not st.session_state.get("authentication_status"):
    st.markdown("""
        <div style="
            text-align: center; 
            padding: 50px 20px; 
            margin: 40px auto; 
            max-width: 800px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        ">
            <h1 style="
                font-size: 5rem; 
                font-weight: 800; 
                margin-bottom: 0; 
                background: linear-gradient(135deg, #f0f4f8 0%, #00d1ff 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: 0 10px 20px rgba(0,0,0,0.2);
            ">BioNexus ML</h1>
            <p style="
                font-size: 1.8rem; 
                color: #d9e2ec; 
                font-weight: 500; 
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-top: 10px;
                opacity: 0.9;
            ">Bioprocess Intelligence Dashboard</p>
            <div style="
                width: 100px; 
                height: 4px; 
                background: #00d1ff; 
                margin: 30px auto 0; 
                border-radius: 2px;
                box-shadow: 0 0 15px rgba(0, 209, 255, 0.5);
            "></div>
        </div>
    """, unsafe_allow_html=True)

st.sidebar.caption("Deployment Version: 3.78-FINAL")
try:
    # Diagnostic Info (Expandable)
    with st.sidebar.expander("🛠️ Diagnostics", expanded=False):
        st.write(f"Users found: {list(config['credentials']['usernames'].keys())}")
        if st.checkbox("Show Emergency Login"):
            secret = st.text_input("Secret Phrase", type="password")
            expected_secret = st.secrets.get("EMERGENCY_SECRET", "bionexus2026")
            if secret == expected_secret:
                if st.button("🚀 Emergency Admin Entry"):
                    st.session_state["authentication_status"] = True
                    st.session_state["username"] = "admin"
                    st.session_state["name"] = "System Admin"
                    st.session_state["_is_emergency"] = True
                    st.rerun()

    # Clear emergency bypass if user gets logged out
    if st.session_state.get("authentication_status") is not True:
        st.session_state["_is_emergency"] = False

    # Skip authenticator if safely in emergency mode
    if not st.session_state.get("_is_emergency", False):
        authenticator.login(location='sidebar', key='login_sidebar')

except Exception as e:
    st.error(e)

if st.session_state.get("authentication_status"):
    username = st.session_state["username"]
    user_info = config["credentials"]["usernames"].get(username)
    
    if user_info:
        if not user_info.get("approved"):
            st.warning("Awaiting admin approval.")
            authenticator.logout(button_name='Logout', location='sidebar', key='logout_approval')
            st.stop()
            
        authenticator.logout(button_name='Logout', location='sidebar', key='logout_main')

        # --- Dashboard Content Starts Here ---
        # Protected Dashboard Logic
        
        # Check if the user is an admin
        user_roles = user_info.get("roles", [])
        user_role = user_info.get("role", "")
        is_admin = "admin" in user_roles or user_role == "admin"

        st.sidebar.markdown("---")
        st.sidebar.write(f"👤 **Active User:** {username}")
        if user_info.get("name"):
            st.sidebar.caption(f"Member: {user_info['name']}")

        if is_admin:
            with st.sidebar.expander("🔑 Admin Management", expanded=True):
                # 1. PENDING APPROVALS
                st.write("#### ⏳ Pending Approvals")
                fresh_config = get_authenticator_config()
                unapproved_users = [u for u, d in fresh_config["credentials"]["usernames"].items() if not d.get("approved", False)]
                
                if not unapproved_users:
                    st.caption("No pending users.")
                else:
                    for u in unapproved_users:
                        st.markdown(f"**{u}**")
                        st.caption(f"{fresh_config['credentials']['usernames'][u]['email']}")
                        if st.button(f"✅ Approve {u}", key=f"appr_pend_{u}", use_container_width=True):
                            update_user_approval(u, True)
                            st.rerun()
                        st.divider()

                # 2. USER CONSOLE
                st.write("#### 👤 User Management")
                users_list = list_users()
                if users_list:
                    for u_data in users_list:
                        u_name = u_data['username']
                        is_self = (u_name == username)
                        
                        # Card-like layout for each user
                        st.markdown(f"**{u_name}** {'(You)' if is_self else ''}")
                        current_role = u_data.get('role') or 'user'
                        st.caption(f"{u_data.get('email', 'N/A')} | {current_role.upper()}")
                        
                        # Use columns for Role and Access toggle only
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            role_val = u_data.get('role') or 'user'
                            new_role = st.selectbox(
                                "Role", options=["user", "admin"], 
                                index=0 if role_val == "user" else 1,
                                key=f"role_{u_name}", disabled=is_self,
                                label_visibility="collapsed"
                            )
                            if new_role != u_data['role']:
                                update_user_role(u_name, new_role)
                                st.rerun()
                        
                        with c2:
                            if u_data['approved']:
                                if st.button("Revoke", key=f"rev_{u_name}", disabled=is_self, use_container_width=True):
                                    update_user_approval(u_name, False)
                                    st.rerun()
                            else:
                                if st.button("Approve", key=f"appr_{u_name}", use_container_width=True):
                                    update_user_approval(u_name, True)
                                    st.rerun()
                        
                        # Delete button in small expander
                        if not is_self:
                            with st.expander("🗑️ Delete"):
                                st.warning("Sure?")
                                if st.button("Confirm", key=f"del_{u_name}", type="primary", use_container_width=True):
                                    delete_user(u_name)
                                    st.rerun()
                        
                        st.divider()
                else:
                    st.info("No users found.")

elif st.session_state.get("authentication_status") is False:
    st.sidebar.error('Invalid credentials')
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.sidebar.warning('Please enter your username and password')

    # Add registration widget
    try:
        # Use fresh config for registration to avoid stale state
        with st.sidebar:
            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(
                roles=['user'],
                merge_username_email=False,
                key='register_user_widget'
            )
        if email_of_registered_user:
            # We must reload and save immediately
            st.sidebar.success('User registered successfully! Awaiting admin approval.')
            
            # The authenticator.register_user might have added it to internal config, 
            # but we need topersist it to our DB.
            # However, stauth handles the hashing? 
            # Actually, register_user in stauth 0.3.x+ returns data and modifies config.
            # We should get the hashed password from the updated config.
            user_data = authenticator.credentials['usernames'][username_of_registered_user]
            
            add_user(
                username=username_of_registered_user,
                email=email_of_registered_user,
                password=user_data['password'],
                name=name_of_registered_user,
                role='user',
                roles=['user'],
                approved=False
            )
    except Exception as e:
        st.sidebar.error(e)
    
    st.stop()

# 3. PROTECTED CONTENT CONTINUES
# Note: Because the entire dashboard below depends on authentication, 
# we wrap it or ensure only authenticated users reach this point.
if not st.session_state.get("authentication_status"):
    st.stop()

# User Context
user_info = config["credentials"]["usernames"].get(username)
user_email = user_info.get('email', '') if user_info else ''

# ----------------------------- Helpers -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    clf = load(model_path)
    return clf

@st.cache_data(show_spinner=False)
def load_schema(schema_path: str) -> dict:
    with open(schema_path, 'r') as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def read_csv(uploaded_file) -> pd.DataFrame:
    if isinstance(uploaded_file, (str, Path)):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)

# --- Shared preprocessing (same as predict.py) ---
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna()

    # Derived features
    if "Glucose_gL" in df.columns:
        df["Glucose_Consumption_Rate"] = df["Glucose_gL"].diff()
    if "Dissolved_Oxygen_percent" in df.columns:
        df["DO_Change"] = df["Dissolved_Oxygen_percent"].diff()
    if {"Product_Titer_gL","Cell_Viability_percent"}.issubset(df.columns):
        df["Specific_Productivity"] = df["Product_Titer_gL"] / df["Cell_Viability_percent"]

    # Normalize agitation
    if "Agitation_RPM" in df.columns:
        scaler = StandardScaler()
        df["Agitation_Normalized"] = scaler.fit_transform(df[["Agitation_RPM"]])

    # Flags
    if "Product_Titer_gL" in df.columns:
        df["High_Titer_Flag"] = (df["Product_Titer_gL"] > 1).astype(int)
    if "Cell_Viability_percent" in df.columns:
        df["Low_Viability_Flag"] = (df["Cell_Viability_percent"] < 98).astype(int)

    return df

def align_columns(df: pd.DataFrame, schema: dict):
    feats = schema['features']
    original_cols = list(df.columns)
    # Add missing
    missing = [c for c in feats if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    # Extra cols
    extra = [c for c in df.columns if c not in feats and c != schema.get('target', 'Product_Titer_gL')]
    # Reorder
    df_aligned = df[feats].copy()
    # Coerce numeric safely
    for c in df_aligned.columns:
        df_aligned.loc[:, c] = pd.to_numeric(df_aligned[c], errors='coerce')
    info = {
        'original_cols': original_cols,
        'expected_features': feats,
        'missing_features': missing,
        'extra_cols_dropped': extra,
    }
    return df_aligned, info

def compute_rmse(y_true, y_pred):
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))

# ----------------------------- Helper Visuals -----------------------------
def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Filter for numeric columns for correlation
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, center=0)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    return fig

def plot_time_series(df: pd.DataFrame, y_cols: list):
    if "Time_hours" not in df.columns:
        st.warning("'Time_hours' column not found for time-series.")
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in y_cols:
        if col in df.columns:
            ax.plot(df["Time_hours"], df[col], marker='o', label=col, alpha=0.7)
    ax.set_title("Process Parameters over Time")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_feature_importance(clf, feature_names):
    # Try to get coefficients or feature importances from the pipeline
    try:
        model_step = clf.named_steps.get('model')
        if not model_step:
             # Try last step if 'model' name is not standard
             model_step = clf.steps[-1][1]
        
        importances = None
        title = "Feature Importance"
        
        if hasattr(model_step, 'coef_'):
            importances = model_step.coef_
            title = "Model Coefficients (Feature Weights)"
        elif hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            title = "Tree-based Feature Importance"
            
        if importances is not None:
            # Flatten in case of multi-output or odd shapes
            importances = np.array(importances).flatten()
            
            # Match with feature names
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis', ax=ax)
            ax.set_title(title)
            plt.tight_layout()
            return fig
    except Exception:
        pass
    return None

# --- Sidebar Theme & Recorder ---
st.sidebar.markdown("---")
st.sidebar.header("🎨 Appearance")
dashboard_theme = st.sidebar.select_slider("Theme", options=["Light", "Dark"], value="Light")
set_design(dashboard_theme, is_authenticated=True)

st.sidebar.markdown("---")
st.sidebar.header("📸 Screen Recorder")
with st.sidebar:
    components.html("""
    <script>
    let mediaRecorder;
    let recordedChunks = [];
    
    function startRec() {
        navigator.mediaDevices.getDisplayMedia({ video: true }).then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);
            mediaRecorder.onstop = () => {
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'bioreactor_dashboard_session.webm';
                a.click();
                recordedChunks = [];
            };
            mediaRecorder.start();
            window.parent.postMessage({type: 'rec_status', status: 'recording'}, '*');
        }).catch(err => {
            alert("Error: " + err);
        });
    }
    
    function stopRec() {
        if (mediaRecorder) {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            window.parent.postMessage({type: 'rec_status', status: 'idle'}, '*');
        }
    }
    </script>
    <div style="display: flex; gap: 10px;">
        <button onclick="startRec()" style="padding: 8px 12px; border-radius: 6px; background: #28a745; color: white; border: none; cursor: pointer; font-weight: 600;">⏺ Start</button>
        <button onclick="stopRec()" style="padding: 8px 12px; border-radius: 6px; background: #dc3545; color: white; border: none; cursor: pointer; font-weight: 600;">⏹ Stop</button>
    </div>
    """, height=50)

# --- Dashboard Content ---
st.title("🔬 Bioreactor ML – Prediction Dashboard")

# Create Tabs
tab_predict, tab_train, tab_explore, tab_history, tab_guide, tab_alerts = st.tabs([
    "🚀 Predict & Benchmark", "🏋️‍♂️ Train Model", "📊 Data Exploration", "📜 History", "📘 Interpretation Guide", "🔔 Alerts"
])

with tab_predict:
    st.write("### Predict results or benchmark model performance")
    st.caption("Use saved sklearn Pipeline model artifacts (.joblib) and schema (.json) to predict Product_Titer_gL from process data.")

    st.sidebar.header("⚙️ Configuration")
    model_choice = st.sidebar.selectbox(
        "Select default model path",
        options=[
            str(ROOT_DIR / 'models/model_ridgecv.joblib'),
            str(ROOT_DIR / 'models/model_gbr.joblib'),
            str(ROOT_DIR / 'models/model_linear.joblib'),
            'Custom...'
        ],
        index=0
    )

    if model_choice == 'Custom...':
        model_path = st.sidebar.text_input("Model path (.joblib)", value=str(ROOT_DIR / 'models/model_ridgecv.joblib'))
    else:
        model_path = model_choice

    schema_path = st.sidebar.text_input("Schema path (.json)", value=str(ROOT_DIR / 'models/feature_schema.json'))

    st.sidebar.markdown("---")
    mode = st.sidebar.radio("Mode", ["Predict (unlabeled)", "Benchmark (labeled)"]) 

    st.sidebar.markdown("---")
    st.sidebar.write("**Actions**")
    if st.sidebar.button("▶️ Run"):
        st.session_state.has_run = True

    # File input
    st.subheader("📥 Input Data")
    upload = st.file_uploader("Upload CSV (new_samples or labeled data)", type=['csv'])
    sample_file_path = ROOT_DIR / 'data/bioreactor_ml_dataset.csv'
    use_sample = st.checkbox(f"Use local sample file: {sample_file_path.name} (if present)")

    input_df = None
    src_desc = None
    if upload is not None:
        input_df = read_csv(upload)
        src_desc = f"Uploaded file: `{upload.name}`"
    elif use_sample and sample_file_path.exists():
        input_df = read_csv(str(sample_file_path))
        src_desc = f"Local sample: `{sample_file_path.name}`"
    else:
        st.info("Upload a CSV or check 'Use local sample file' to proceed.")

    if input_df is not None:
        # Store in session state for tab_explore
        st.session_state.input_df = input_df
        with st.expander("Preview input data", expanded=False):
            st.write(src_desc)
            st.dataframe(input_df.head(20), use_container_width=True)
            st.write(f"Rows: {len(input_df):,}  |  Columns: {len(input_df.columns)}")

    # ----------------------------- Run -----------------------------
    if st.session_state.get("has_run") and input_df is not None:
        try:
            clf = load_model(model_path)
            schema = load_schema(schema_path)
            target_col = schema.get('target', 'Product_Titer_gL')

            # Preprocess
            proc_df = input_df.copy()
            proc_df = preprocess(proc_df)

            # Align columns
            X_aligned, align_info = align_columns(proc_df.copy(), schema)
            with st.expander("Schema alignment details", expanded=False):
                st.write("**Expected features:**", align_info['expected_features'])
                st.write("**Missing features auto-added (NaN):**", align_info['missing_features'])
                st.write("**Extra columns dropped:**", align_info['extra_cols_dropped'])

            # Prediction or Benchmark logic
            if mode == "Predict (unlabeled)":
                preds = clf.predict(X_aligned)
                out = input_df.copy()
                # Use pd.Series to auto-align based on index (handles dropped rows as NaN)
                out['Pred_Product_Titer_gL'] = pd.Series(preds, index=X_aligned.index)
                st.success("Predictions completed.")
                
                # --- AUTO SAVE HISTORY ---
                save_prediction(
                    username=username,
                    inputs={"file": src_desc, "rows": len(input_df), "mode": "Predict"},
                    results={"mean_pred": float(np.mean(preds)), "count": len(preds)},
                    model_name=model_path
                )
                
                # --- CHECK ALERTS ---
                alert_cfg = get_alert_config(username)
                if alert_cfg.get('email_enabled'):
                    mean_val = float(np.mean(preds))
                    threshold = alert_cfg.get('titer_threshold', 5.0)
                    condition = alert_cfg.get('condition', 'above')
                    
                    trigger = (condition == 'above' and mean_val > threshold) or \
                              (condition == 'below' and mean_val < threshold)
                    
                    if trigger:
                        subject = f"🚨 BiNexus Alert: Prediction Threshold Met ({username})"
                        body = f"Hello {username},\n\nYour recent batch prediction (mean: {mean_val:.2f} g/L) met the alert condition '{condition} {threshold} g/L'.\n\nModel: {model_path}\nRows: {len(preds)}"
                        # Send in background thread
                        Thread(target=send_email_alert, args=(alert_cfg.get('target_email'), subject, body, alert_cfg)).start()
                        st.toast("📧 Alert email triggered!")

                st.toast("✅ Batch prediction saved to history")
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.write("#### Results Preview (Top 100)")
                    st.dataframe(out.head(100), use_container_width=True)
                    csv_bytes = out.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download predictions.csv", data=csv_bytes,
                                    file_name="predictions.csv", mime="text/csv")
                with c2:
                    st.write("#### Distribution")
                    fig, ax = plt.subplots(figsize=(6,6))
                    sns.histplot(preds, bins=20, kde=True, ax=ax, color='#0054A0')
                    ax.set_title('Predicted Distribution')
                    st.pyplot(fig)
                    with st.expander("💡 How to read this distribution?"):
                        st.info("""
                        - **Peak Location**: The most likely yield for your current batch.
                        - **Spread (Width)**: A narrow peak means high confidence; a wide spread suggests the process is sensitive to small parameter shifts.
                        - **Skewness**: If leaning right, your process conditions are optimized for high yield.
                        """)
            else:
                if target_col not in input_df.columns:
                    st.error(f"Target column '{target_col}' not found in provided CSV.")
                else:
                    # Deriving from proc_df ensures alignment with X_aligned (both processed)
                    y_series = pd.to_numeric(proc_df[target_col], errors='coerce')
                    valid_mask = y_series.notnull()
                    y_true = y_series[valid_mask]
                    X_eval = X_aligned[valid_mask]
                    
                    if len(y_true) == 0:
                        st.error("No valid numeric target values found in the target column after preprocessing.")
                    else:
                        preds = clf.predict(X_eval)
                        r2 = r2_score(y_true, preds)
                        mae = mean_absolute_error(y_true, preds)
                        rmse = compute_rmse(y_true, preds)
                        
                        st.write("#### 📊 Performance Metrics")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Test R²", f"{r2:.4f}")
                        c2.metric("MAE (g/L)", f"{mae:.4f}")
                        c3.metric("RMSE (g/L)", f"{rmse:.4f}")

                        # --- AUTO SAVE HISTORY ---
                        save_prediction(
                            username=username,
                            inputs={"file": src_desc, "rows": len(y_true), "mode": "Benchmark"},
                            results={"R2": r2, "MAE": mae, "RMSE": rmse},
                            model_name=model_path
                        )

                        # --- CHECK ALERTS ---
                        alert_cfg = get_alert_config(username)
                        if alert_cfg.get('email_enabled'):
                            # For benchmark, use R2 or RMSE? Let's stick to yield-based logic if possible, 
                            # or just skip for benchmark unless user asks.
                            # Usually alerts are for new predictions.
                            pass

                        st.toast("✅ Benchmark run saved to history")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("#### 🎯 Actual vs Predicted")
                            fig, ax = plt.subplots(figsize=(6,6))
                            ax.scatter(y_true, preds, alpha=0.6, color='darkblue')
                            lims = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
                            ax.plot(lims, lims, 'r--', label='Ideal')
                            ax.set_xlabel('Actual')
                            ax.set_ylabel('Predicted')
                            ax.legend()
                            st.pyplot(fig)
                        
                        with col_b:
                            st.write("#### 💡 Model Insights")
                            feat_fig = plot_feature_importance(clf, schema['features'])
                            if feat_fig:
                                st.pyplot(feat_fig)
                                with st.expander("💡 What are CPPs?"):
                                    st.write("""
                                    Items at the top are your **Critical Process Parameters**. 
                                    - **Positive Bar**: Increasing this value (e.g., Temp) likely increases your yield.
                                    - **Negative Bar**: Increasing this value might inhibit growth or production.
                                    - **Small Bar**: The model finds this parameter less relevant for the current prediction.
                                    """)
                            else:
                                st.info("Feature importance not available for this model type.")

                        with st.expander("Residuals Analysis"):
                            residuals = y_true - preds
                            fig, ax = plt.subplots(figsize=(10,4))
                            sns.histplot(residuals, bins=30, kde=True, ax=ax, color='#00A1DE')
                            ax.axvline(0, color='red', linestyle='--')
                            ax.set_title('Residuals Distribution (Actual - Predicted)')
                            st.pyplot(fig)

        except Exception as e:
            st.exception(e)

with tab_train:
    st.write("### 🏋️‍♂️ Train a New Model")
    st.caption("Upload labeled data to train a new RidgeCV pipeline. This will save a `.joblib` model and a `.json` schema.")
    
    train_upload = st.file_uploader("Upload Training CSV", type=['csv'], key="train_up")
    
    c1, c2 = st.columns(2)
    with c1:
        target_name = st.text_input("Target Column", value="Product_Titer_gL")
        out_model_name = st.text_input("Output Model Path", value="outputs/model_custom.joblib")
    with c2:
        out_schema_name = st.text_input("Output Schema Path", value="outputs/feature_schema_custom.json")
    
    if st.button("🔥 Start Training"):
        if train_upload is not None:
            try:
                with st.spinner("Preprocessing and training..."):
                    raw_train_df = pd.read_csv(train_upload)
                    if target_name not in raw_train_df.columns:
                        st.error(f"Target '{target_name}' not found.")
                    else:
                        # 1. Preprocess
                        df_processed = preprocess(raw_train_df)
                        
                        # 2. Train
                        X_train = df_processed.drop(columns=[target_name])
                        y_train = df_processed[target_name]
                        
                        pipe = Pipeline([
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler()),
                            ('model', RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5))
                        ])
                        pipe.fit(X_train, y_train)
                        
                        # 3. Save
                        Path("outputs").mkdir(exist_ok=True)
                        dump(pipe, out_model_name)
                        
                        schema = {
                            'target': target_name,
                            'features': X_train.columns.tolist(),
                            'dtypes': {c: str(df_processed[c].dtype) for c in X_train.columns},
                            'trained_rows': int(df_processed.shape[0]),
                            'trained_cols': int(df_processed.shape[1])
                        }
                        with open(out_schema_name, 'w') as f:
                            json.dump(schema, f, indent=2)
                        
                        st.success(f"✅ Model trained and saved to `{out_model_name}`")
                        
                        # 4. Results
                        train_preds = pipe.predict(X_train)
                        r2_train = r2_score(y_train, train_preds)
                        st.metric("Training R²", f"{r2_train:.4f}")
            except Exception as e:
                st.exception(e)
        else:
            st.warning("Please upload a training dataset first.")

with tab_explore:
    explore_df = st.session_state.get("input_df")
    
    if explore_df is None:
        st.info("Please upload data in the 'Predict' tab to begin exploration.")
    else:
        st.write("### 🔍 Process Data Exploration")
        
        m1, m2, m3, m4 = st.columns(4)
        if "Temperature_C" in explore_df.columns:
            m1.metric("Avg Temp", f"{explore_df['Temperature_C'].mean():.1f} °C")
        if "pH" in explore_df.columns:
            m2.metric("Avg pH", f"{explore_df['pH'].mean():.2f}")
        if "Dissolved_Oxygen_percent" in explore_df.columns:
            m3.metric("Avg DO", f"{explore_df['Dissolved_Oxygen_percent'].mean():.1f} %")
        if "Product_Titer_gL" in explore_df.columns:
            m4.metric("Avg Titer", f"{explore_df['Product_Titer_gL'].mean():.2f} g/L")

        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### 🔥 Parameter Correlations")
            heat_fig = plot_correlation_heatmap(explore_df)
            st.pyplot(heat_fig)
            
        with col2:
            st.write("#### 📈 Time-Series Trends")
            all_cols = list([c for c in explore_df.columns if c != "Time_hours"])
            selected_metrics = st.multiselect(
                "Select parameters",
                options=all_cols,
                default=["Temperature_C", "pH"] if "Temperature_C" in all_cols and "pH" in all_cols else (all_cols[:2] if len(all_cols) >= 2 else all_cols)
            )
            if selected_metrics:
                ts_fig = plot_time_series(explore_df, selected_metrics)
                if ts_fig:
                    st.pyplot(ts_fig)

        st.markdown("---")
        st.write("#### 📊 Distributions")
        numeric_cols = explore_df.select_dtypes(include=[np.number]).columns
        dist_col = st.selectbox("Select feature", options=numeric_cols, key="dist_sel")
        
        c_hist, c_box = st.columns([2, 1])
        with c_hist:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(explore_df[dist_col], kde=True, ax=ax, color='teal')
            st.pyplot(fig)
        with c_box:
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.boxplot(y=explore_df[dist_col], ax=ax, color='lightgreen')
            st.pyplot(fig)

with tab_history:
    st.write("### 📜 Prediction History")
    st.caption("View and manage your past model runs and benchmark results.")
    
    history = get_user_history(username)
    
    if not history:
        st.info("No history found. Try running a prediction in the 'Predict' tab.")
    else:
        # Prepare table data
        table_data = []
        for item in history:
            res = item['results']
            summary = ""
            if 'R2' in res:
                summary = f"R²: {res['R2']:.3f} | RMSE: {res['RMSE']:.3f}"
            elif 'mean_pred' in res:
                summary = f"Mean: {res['mean_pred']:.2f} | Rows: {res['count']}"
                
            table_data.append({
                "ID": item['id'],
                "Timestamp": item['timestamp'],
                "Model": item['model_name'].split('/')[-1],
                "Summary": summary,
                "Mode": item['inputs'].get('mode', 'N/A')
            })
        
        hist_df = pd.DataFrame(table_data)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        
        # Actions
        st.write("#### Detailed Actions")
        item_id = st.selectbox("Select Run ID to Inspect/Delete", options=[d['ID'] for d in table_data])
        
        col_view, col_del = st.columns(2)
        selected_item = next((h for h in history if h['id'] == item_id), None)
        
        if selected_item:
            with col_view:
                if st.button("🔍 View Details", use_container_width=True):
                    st.json(selected_item)
            
            with col_del:
                if st.button("🗑️ Delete from History", use_container_width=True, type="secondary"):
                    delete_history_item(item_id, username)
                    st.toast(f"Deleted run {item_id}")
                    st.rerun()
        
        # Download all history
        full_df = pd.DataFrame(history)
        csv_data = full_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full History (CSV)",
            data=csv_data,
            file_name=f"history_{username}.csv",
            mime='text/csv',
            use_container_width=True
        )

with tab_guide:
    st.write("## 📘 Bioprocessor's Analytics Masterclass")
    st.markdown("""
    Welcome to the deep-dive guide. This section bridges the gap between **Data Science** and **Bioprocess Engineering**, helping you turn graphs into actionable lab decisions.
    """)
    
    # --- Section 1: Accuracy & Trust ---
    with st.expander("🎯 1. Model Performance (Actual vs. Predicted)", expanded=True):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 🔬 The Science")
            st.write("""
            In biology, "Noise" is constant. A model that tracks every dot perfectly is often **overfit** (modeling random error). We look for a model that captures the **unlying metabolic trend** despite sensor fluctuations.
            """)
            st.markdown("#### 📊 How to Read")
            st.write("""
            - **The Identity Line (Dash)**: This is "Truth". The closer your dots cluster here, the more the model understands your process.
            - **Bias Check**: If all dots are *below* the line at high yields, your model is "Pessimistic" and under-predicting your best batches.
            """)
        with col2:
            st.markdown("#### 🛠️ Action Plan")
            st.warning("""
            - **Low R² (<0.6)**: Your inputs (Temp, pH, etc.) don't explain the yield enough. Look for unmeasured factors like *Media Lot Variation* or *Inoculum Quality*.
            - **High RMSE**: If your RMSE is 1.0 on a 5.0 yield, that's a 20% error. Do not use this model for critical setpoint changes yet.
            """)

    # --- Section 2: Drivers & Inhibitors ---
    with st.expander("💡 2. Critical Process Parameters (Feature Importance)"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 🔬 The Science")
            st.write("""
            Bioreactors are non-linear. A small change in Temperature might do nothing at 30°C but cause a "metabolic crash" at 38°C. Importance shows which variable is the **limiting factor** for production.
            """)
            st.markdown("#### 📊 How to Read")
            st.write("""
            - **Top Bars**: These are your "Levers". Changing these will have the biggest impact on your yield.
            - **Negative Importance**: Increasing this variable actually *hurts* your final titer.
            """)
        with col2:
            st.markdown("#### 🛠️ Action Plan")
            st.success("""
            - **Optimization**: If "Agitation" is top importance and positive, your process is **Oxygen Limited**. Increase your RPM or Sparger rate to unlock more growth.
            - **Stability**: If "pH" has high negative importance, your control loop is too wide. Tighten your acid/base pump settings.
            """)

    # --- Section 3: Coupled Dynamics ---
    with st.expander("🔥 3. Parameter Correlations (The Heatmap)"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 🔬 The Science")
            st.write("""
            Variables in a tank are rarely independent. Henry's Law dictates that as Temperature rises, Oxygen solubility drops.
            """)
            st.markdown("#### 📊 How to Read")
            st.write("""
            - **Deep Red (+1.0)**: Perfect coupling. If these two always move together, you might only need to measure one of them to save costs.
            - **Deep Blue (-1.0)**: Inverse relationship. When cells grow (Biomass ⭡), they consume nutrients (Glucose ⭣).
            """)
        with col2:
            st.markdown("#### 🛠️ Action Plan")
            st.info("""
            - **Sensor Health**: If two variables that *should* correlate (like Agitation and DO) show 0 correlation, check for a fouled DO probe or a snapped impeller shaft.
            """)

    # --- Section 4: QC & Outliers ---
    with st.expander("📊 4. Distribution & Stability (Boxplots)"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 🔬 The Science")
            st.write("""
            Consistency is the goal of GMP manufacturing. Distributions show you the "Natural Variation" of your process across multiple runs.
            """)
            st.markdown("#### 📊 How to Read")
            st.write("""
            - **The Box**: Represents the 50% "Expected" range.
            - **Whiskers/Dots**: These are your **Outliers**. These represent batches where something went "Wrong" (Contamination, Power fail, etc.).
            """)
        with col2:
            st.markdown("#### 🛠️ Action Plan")
            st.error("""
            - **Root Cause Analysis**: Click on an outlier ID in the History tab. See what parameter was unique during that run to prevent future batch losses.
            """)

    st.markdown("---")
    st.write("### 📖 Glossary of Metrics")
    c1, c2, c3 = st.columns(3)
    c1.help("**R-Squared (R²)**: The percentage of yield variation explained by your inputs. 0.8+ is excellent.")
    c2.help("**MAE**: Mean Absolute Error. The 'Average Miss' in real units (g/L).")
    c3.help("**RMSE**: Penalizes large errors more heavily than MAE. High RMSE means the model sometimes makes very big mistakes.")

with tab_alerts:
    st.write("## 🔔 Smart Email Alerts")
    st.caption("Configure automated notifications based on model prediction results.")
    
    alert_cfg = get_alert_config(username)
    
    with st.form("alert_config_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 📢 Notification Rules")
            enabled = st.toggle("🔔 Enable Email Alerts", value=bool(alert_cfg.get('email_enabled')))
            target_email = st.text_input("📧 Recipient Email", value=alert_cfg.get('target_email') or user_email)
            
            st.write("### 🎯 Criteria")
            threshold = st.number_input("📏 Titer Threshold (g/L)", value=float(alert_cfg.get('titer_threshold', 5.0)), step=0.1)
            condition = st.selectbox("⚖️ Trigger Condition", options=["above", "below"], index=0 if alert_cfg.get('condition') == 'above' else 1)
        
        with col2:
            st.write("### 🔐 SMTP Settings")
            st.info("💡 Connectivity details required for outbound emails. Use an App Password for Gmail.")
            server = st.text_input("🌐 SMTP Server", value=alert_cfg.get('smtp_server') or "smtp.gmail.com")
            port = st.number_input("🔌 SMTP Port", value=int(alert_cfg.get('smtp_port', 587)))
            user = st.text_input("👤 SMTP Username", value=alert_cfg.get('smtp_user') or "")
            password = st.text_input("🔑 SMTP Password", value=alert_cfg.get('smtp_pass') or "", type="password")
            
        submitted = st.form_submit_button("💾 SAVE SETTINGS", use_container_width=True)
        if submitted:
            new_cfg = {
                'email_enabled': enabled,
                'target_email': target_email,
                'titer_threshold': threshold,
                'condition': condition,
                'smtp_server': server,
                'smtp_port': port,
                'smtp_user': user,
                'smtp_pass': password
            }
            save_alert_config(username, new_cfg)
            st.success("Alert configuration updated!")
            st.rerun()

    st.markdown("---")
    st.write("### 🧪 Connectivity Check")
    if st.button("📤 SEND TEST EMAIL", use_container_width=True):
        if not alert_cfg.get('smtp_user') or not alert_cfg.get('smtp_pass'):
            st.error("❌ Please configure and save SMTP settings first!")
        else:
            with st.spinner("🚀 Dispatching test email..."):
                success = send_email_alert(
                    alert_cfg.get('target_email'),
                    "🧪 BioNexus SMTP Test",
                    f"Hello {username},\n\nYour BioNexus Dashboard is now configured for active monitoring! Your SMTP settings are valid.",
                    alert_cfg
                )
                if success:
                    st.success(f"✅ Connection successful! Test email sent to {alert_cfg.get('target_email')}")
                else:
                    st.error("❌ Connection failed. Check your App Password or Server settings.")
