
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide"
)

# -----------------------------
# VIDEO BACKGROUND
# -----------------------------
def add_bg_video(video_file):
    try:
        with open(video_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(f"""
<style>

[data-testid="stAppViewContainer"]{{
    background: transparent !important;
}}

.main{{
    background: transparent !important;
}}

#bgvideo{{
    position: fixed;
    top: 0;
    left: 0;
    bottom:50%;                
    min-width: 100vw;
    min-height: 90vh;
    object-fit: fully;
    z-index: -999;
    opacity: 0.85;
    filter:brightness(100%);
}}

[data-testid="stAppViewContainer"] {{
    background: transparent !important;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stSidebar"] {{
    background: #000000 !important;
    border-right: 2px solid #00d4ff;
}}

[data-testid="stSidebar"] * {{
    color: white !important;
}}

.block-container {{
    background: rgba(0,0,0,0.35);
    backdrop-filter: blur(5px);
    border-radius: 20px;
    padding: 2rem;
}}

h1,h2,h3,h4,p,label {{
    color: white !important;
}}
                    
.stTextInput > div > div > input{{
    background-color: rgba(0,0,0,0.75) !important;
    color: white !important;
    border: 2px solid #00d4ff !important;
    border-radius: 10px;
}}

label{{
    color:white !important;
}}

.login-box{{
    background: rgba(0,0,0,0.55);
    padding:30px;
    border-radius:20px;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.2);
}}
[data-testid="stSidebar"]{{
    background:
        linear-gradient(
            rgba(0,0,0,0.80),
            rgba(0,0,0,0.90)
        ) !important;
}}
[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0.88) !important;
    border-right: 2px solid #00d4ff;
}}
div.stButton > button {{
    background: #00d4ff !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 10px;
}}
[data-testid="stSidebar"] * {{
    color: white !important;
   #bgvideo {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        z-index: -9999;
        opacity: 0.35;
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}

    [data-testid="stHeader"] {{
        background: rgba(5,5,5,5) !important;
    }}

    .main {{
        background: transparent !important;
    }}

    .block-container {{
        background: rgba(0,0,0,0.25);
        backdrop-filter: blur(4px);
        border-radius: 20px;
        padding: 2rem;
    }}
    /* Main app background */
.stApp {{
    background: transparent !important;
}}

/* Main content area */
[data-testid="stAppViewContainer"] {{
    background: transparent !important;
}}

/* Header */
[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* Main section */
[data-testid="stMain"] {{
    background: transparent !important;
}}

/* Block container */
.block-container {{
    background: transparent !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0.85) !important;
}}

/* Make text visible */
h1, h2, h3, h4, h5, h6, p, label, span {{
    color: white !important;

}}
[data-testid="stSidebar"]{{
    background: linear-gradient(
        180deg,
        #0f172a 50%,
        #1e3a8a 50%,
        #2563eb 100%
    ) !important;
}}

[data-testid="stSidebar"] *{{
    color: Blue !important;
    font-weight: 500;
}}
</style>

<video autoplay muted loop id="bgvideo">
    <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
</video>

""", unsafe_allow_html=True)
    except Exception:
        pass


add_bg_video("factory.mp4")

# -----------------------------
# LOGIN
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.markdown(
        "<h1 style='text-align:center'>🏭 Predictive Maintenance Dashboard</h1>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1,2,1])

    with c2:
        st.subheader("Login")

        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if user == "admin" and pwd == "admin123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def train_model(data):

    X = data[[
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]]

    y = data["Target"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"])
        ],
        remainder="passthrough"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)

    return pipe, acc

model, accuracy = train_model(df)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.title("🏭 Dashboard")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Prediction",
            "Dataset"
        ]
    )

    st.markdown("---")

    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    st.markdown("---")

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# -----------------------------
# OVERVIEW
# -----------------------------
if page == "Overview":

    st.title("🏭 Predictive Maintenance System")

    total_records = len(df)
    failures = int(df["Target"].sum())

    failure_rate = round(
        (failures / total_records) * 100,
        2
    )

    health_score = round(
        100 - failure_rate,
        2
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", total_records)
    c2.metric("Failures", failures)
    c3.metric("Failure Rate", f"{failure_rate}%")
    c4.metric("Health Score", f"{health_score}%")

    st.divider()

    st.subheader("Machine Type Distribution")

    fig = px.histogram(
        df,
        x="Type",
        color="Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:

        fail_counts = df["Target"].value_counts()

        fig1 = px.pie(
            values=fail_counts.values,
            names=["Healthy", "Failure"]
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        failure_types = (
            df[df["Target"] == 1]["Failure Type"]
            .value_counts()
            .reset_index()
        )

        failure_types.columns = [
            "Failure Type",
            "Count"
        ]

        fig2 = px.bar(
            failure_types,
            x="Failure Type",
            y="Count"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("RPM vs Torque")

    fig3 = px.scatter(
        df,
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        color="Target"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Temperature Analysis")

    fig4 = px.scatter(
        df,
        x="Air temperature [K]",
        y="Process temperature [K]",
        color="Target"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Correlation Heatmap")

    numeric_cols = df.select_dtypes(include=np.number)

    corr = numeric_cols.corr()

    fig5 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Turbo"
    )

    st.plotly_chart(fig5, use_container_width=True)

    if health_score > 90:
        st.success("🟢 Factory operating normally")
    elif health_score > 75:
        st.warning("🟡 Preventive maintenance recommended")
    else:
        st.error("🔴 Immediate maintenance required")

# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "Prediction":

    st.title("🔮 Failure Prediction")

    col1, col2 = st.columns(2)

    with col1:

        machine_type = st.selectbox(
            "Type",
            ["L", "M", "H"]
        )

        air_temp = st.number_input(
            "Air temperature [K]",
            value=300.0
        )

        process_temp = st.number_input(
            "Process temperature [K]",
            value=310.0
        )

    with col2:

        rpm = st.number_input(
            "Rotational speed [rpm]",
            value=1500
        )

        torque = st.number_input(
            "Torque [Nm]",
            value=40.0
        )

        tool_wear = st.number_input(
            "Tool wear [min]",
            value=10
        )

    if st.button("Predict Failure", use_container_width=True):

        sample = pd.DataFrame({
            "Type":[machine_type],
            "Air temperature [K]":[air_temp],
            "Process temperature [K]":[process_temp],
            "Rotational speed [rpm]":[rpm],
            "Torque [Nm]":[torque],
            "Tool wear [min]":[tool_wear]
        })

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]

        st.metric(
            "Failure Probability",
            f"{probability*100:.2f}%"
        )

        if prediction == 1:
            st.error("🔴 Failure Risk Detected")
        else:
            st.success("🟢 Machine Appears Healthy")

# -----------------------------
# DATASET PAGE
# -----------------------------
else:

    st.title("📊 Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode()

    st.download_button(
        "Download Dataset",
        csv,
        "predictive_maintenance.csv",
        "text/csv"
    )
