import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        border: 1px solid #e8edf4;
    }
    .small-note {color: #64748b; font-size: 0.85rem;}

    .brand-header {
        display:flex; align-items:center; gap:16px;
        padding:18px 22px; margin:0 0 22px 0;
        border-radius:18px;
        background:linear-gradient(110deg,#08284d 0%,#124b87 72%,#d71920 145%);
        box-shadow:0 10px 28px rgba(8,40,77,.18);
        color:#fff;
    }
    .brand-mark {
        width:66px; height:66px; flex:0 0 66px;
        display:flex; align-items:center; justify-content:center;
        border-radius:18px;
        background:linear-gradient(145deg,#ffffff,#dcecff);
        box-shadow:inset 0 1px 5px rgba(255,255,255,.8),0 5px 14px rgba(0,0,0,.18);
    }
    .brand-title {font-size:1.72rem;font-weight:800;line-height:1.15;letter-spacing:-.5px;color:#fff;}
    .brand-title span {color:#ff6b70;}
    .brand-subtitle {font-size:.91rem;color:#e5efff;margin-top:5px;}
    .brand-chip {
        margin-left:auto; padding:8px 13px; border-radius:999px;
        border:1px solid rgba(255,255,255,.28); background:rgba(255,255,255,.10);
        color:#f5f8ff; font-size:.78rem; font-weight:700; white-space:nowrap;
    }
    @media(max-width:700px) {
        .brand-chip {display:none;}
        .brand-title {font-size:1.25rem;}
        .brand-mark {width:54px;height:54px;flex-basis:54px;}
    }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand-header">
  <div class="brand-mark">
    <svg width="48" height="48" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M9 48V35" stroke="#174A8B" stroke-width="7" stroke-linecap="round"/>
      <path d="M23 48V25" stroke="#174A8B" stroke-width="7" stroke-linecap="round"/>
      <path d="M37 48V16" stroke="#174A8B" stroke-width="7" stroke-linecap="round"/>
      <path d="M7 19L22 12L34 19L53 5" stroke="#D71920" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M44 5H53V14" stroke="#D71920" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="13" cy="53" r="4" fill="#16845B"/>
      <circle cx="27" cy="53" r="4" fill="#16845B"/>
      <circle cx="41" cy="53" r="4" fill="#16845B"/>
    </svg>
  </div>
  <div>
    <div class="brand-title">Customer <span>Churn</span> Prediction</div>
    <div class="brand-subtitle">Predict • Analyze • Retain &nbsp; | &nbsp; Machine Learning Customer Intelligence</div>
  </div>
  <div class="brand-chip">✦ SMART RETENTION ANALYTICS</div>
</div>
""", unsafe_allow_html=True)

# -------------------- DATA + MODEL TRAINING --------------------

@st.cache_data
def load_data(file_bytes):
    from io import BytesIO
    data = pd.read_csv(BytesIO(file_bytes))
    return data

@st.cache_resource
def train_models(file_bytes):
    from io import BytesIO
    raw = pd.read_csv(BytesIO(file_bytes))

    required = {"Churn"}
    if not required.issubset(raw.columns):
        raise ValueError("Dataset must contain a 'Churn' column.")

    data = raw.copy()
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    if data["Churn"].isna().any():
        raise ValueError("Churn column must contain only Yes/No values.")

    X = data.drop(columns=["Churn"])
    y = data["Churn"].astype(int)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="drop"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced", max_depth=8
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=42,
            class_weight="balanced", n_jobs=-1
        )
    }

    fitted = {}
    rows = []
    predictions = {}
    probabilities = {}

    for name, estimator in base_models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        fitted[name] = pipe
        predictions[name] = pred
        probabilities[name] = proba

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba)
        })

    metrics = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    best_name = metrics.iloc[0]["Model"]

    return {
        "data": data,
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test": y_test,
        "models": fitted,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
        "best_name": best_name,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }

# -------------------- SIDEBAR / DATASET --------------------

st.sidebar.title("⚙️ Dashboard Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload Telco-Customer-Churn.csv",
    type=["csv"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
else:
    st.sidebar.info("Using the default CSV file from the project folder.")
    try:
        with open("Telco-Customer-Churn.csv", "rb") as f:
            file_bytes = f.read()
    except FileNotFoundError:
        st.error(
            "Dataset not found. Place 'Telco-Customer-Churn.csv' in the same "
            "folder as app.py, or upload it using the sidebar."
        )
        st.stop()

try:
    result = train_models(file_bytes)
except Exception as exc:
    st.error(f"Could not prepare the dataset/models: {exc}")
    st.stop()

data = result["data"]
X = result["X"]
y = result["y"]
metrics = result["metrics"]
best_name = result["best_name"]
best_model = result["models"][best_name]

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Churn Analytics", "Model Comparison", "Customer Prediction", "Feature Importance"]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Best model by test ROC-AUC: **{best_name}**")
st.sidebar.caption("Models are trained locally when the dataset changes.")

# -------------------- OVERVIEW --------------------

if page == "Overview":
    st.subheader("Business Overview")
    total_customers = len(data)
    churned = int((data["Churn"] == 1).sum())
    retained = int((data["Churn"] == 0).sum())
    churn_rate = (churned / total_customers * 100) if total_customers else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_rate:.1f}%")

    left, right = st.columns(2)
    with left:
        churn_counts = data["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().rename_axis("Status").reset_index(name="Customers")
        fig = px.pie(
            churn_counts, names="Status", values="Customers",
            hole=0.55, title="Customer Churn Distribution",
            color="Status",
            color_discrete_map={"Stayed": "#16a34a", "Churned": "#ef4444"}
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if "Contract" in data.columns:
            contract_churn = (
                data.assign(ChurnLabel=data["Churn"].map({0: "Stayed", 1: "Churned"}))
                .groupby(["Contract", "ChurnLabel"]).size().reset_index(name="Customers")
            )
            fig = px.bar(
                contract_churn, x="Contract", y="Customers", color="ChurnLabel",
                barmode="group", title="Churn by Contract Type",
                color_discrete_map={"Stayed": "#16a34a", "Churned": "#ef4444"}
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)
    st.caption(f"Dataset contains {data.shape[0]:,} rows and {data.shape[1]:,} columns, including the target column.")

# -------------------- CHURN ANALYTICS --------------------

elif page == "Churn Analytics":
    st.subheader("Customer Churn Analytics")
    col1, col2 = st.columns(2)

    with col1:
        if "tenure" in data.columns:
            fig = px.histogram(
                data, x="tenure", color=data["Churn"].map({0: "Stayed", 1: "Churned"}),
                barmode="overlay", opacity=0.75, nbins=30,
                title="Tenure Distribution by Churn",
                labels={"color": "Customer Status"},
                color_discrete_map={"Stayed": "#16a34a", "Churned": "#ef4444"}
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "MonthlyCharges" in data.columns:
            fig = px.box(
                data, x=data["Churn"].map({0: "Stayed", 1: "Churned"}),
                y="MonthlyCharges", color=data["Churn"].map({0: "Stayed", 1: "Churned"}),
                title="Monthly Charges vs Churn",
                labels={"x": "Customer Status", "color": "Customer Status"},
                color_discrete_map={"Stayed": "#16a34a", "Churned": "#ef4444"}
            )
            st.plotly_chart(fig, use_container_width=True)

    categorical_candidates = [
        c for c in ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen",
                    "PaperlessBilling", "TechSupport", "OnlineSecurity"] if c in data.columns
    ]
    selected = st.selectbox("Explore churn by customer attribute", categorical_candidates)
    grouped = (
        data.groupby(selected, dropna=False)["Churn"]
        .mean().mul(100).reset_index(name="Churn Rate (%)")
    )
    grouped[selected] = grouped[selected].astype(str)
    fig = px.bar(
        grouped, x=selected, y="Churn Rate (%)",
        title=f"Churn Rate by {selected}", text_auto=".1f"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- MODEL COMPARISON --------------------

elif page == "Model Comparison":
    st.subheader("Machine Learning Model Comparison")
    st.caption("All metrics below are calculated on the held-out test set.")

    st.dataframe(
        metrics.style.format({
            "Accuracy": "{:.3f}", "Precision": "{:.3f}",
            "Recall": "{:.3f}", "F1 Score": "{:.3f}", "ROC-AUC": "{:.3f}"
        }),
        use_container_width=True
    )

    metric_choice = st.selectbox(
        "Choose metric to compare",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    )
    fig = px.bar(
        metrics, x="Model", y=metric_choice, color="Model",
        title=f"Model Comparison - {metric_choice}",
        text_auto=".3f", range_y=[0, 1]
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    y_test = result["y_test"]
    for name in result["models"]:
        fpr, tpr, _ = roc_curve(y_test, result["probabilities"][name])
        auc_value = roc_auc_score(y_test, result["probabilities"][name])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={auc_value:.3f})"
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random baseline", line=dict(dash="dash", color="gray")
    ))
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(range=[0, 1]), xaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Best model by ROC-AUC:** {best_name}")

    cm = confusion_matrix(
        result["y_test"], result["predictions"][best_name], labels=[0, 1]
    )
    cm_fig = px.imshow(
        cm, text_auto=True, x=["Predicted Stayed", "Predicted Churned"],
        y=["Actual Stayed", "Actual Churned"],
        color_continuous_scale="Blues", title=f"{best_name} - Confusion Matrix"
    )
    st.plotly_chart(cm_fig, use_container_width=True)

# -------------------- CUSTOMER PREDICTION --------------------

elif page == "Customer Prediction":
    st.subheader("Predict Churn for an Individual Customer")
    st.write("Enter customer details below. The selected model will estimate the churn probability.")

    with st.form("customer_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        with col3:
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=2000.0, value=70.0, step=1.0)
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=float(monthly * tenure), step=10.0)

        submitted = st.form_submit_button("🔍 Predict Customer Churn", use_container_width=True)

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        # Align input columns with the exact training feature order.
        customer = customer.reindex(columns=X.columns)

        probability = float(best_model.predict_proba(customer)[0, 1])
        prediction = int(probability >= 0.5)

        st.markdown("---")
        st.markdown("### Prediction Result")
        a, b, c = st.columns(3)
        a.metric("Predicted Status", "Likely to Churn" if prediction else "Likely to Stay")
        b.metric("Churn Probability", f"{probability * 100:.1f}%")

        if probability < 0.30:
            risk = "Low"
            advice = "Continue regular engagement and monitor customer satisfaction."
            color = "success"
        elif probability < 0.60:
            risk = "Medium"
            advice = "Review service usage and consider a suitable loyalty benefit."
            color = "warning"
        else:
            risk = "High"
            color = "error"
            advice = "Prioritize a customer-service follow-up and review contract or service concerns."

        c.metric("Risk Category", risk)
        st.progress(probability)
        st.write(f"**Suggested retention action:** {advice}")
        st.caption(
            f"Prediction generated using {best_name}. Risk bands are illustrative project rules, "
            "not calibrated business policy. Validate them before real-world use."
        )

# -------------------- FEATURE IMPORTANCE --------------------

elif page == "Feature Importance":
    st.subheader("Feature Importance")
    rf_pipe = result["models"]["Random Forest"]
    rf = rf_pipe.named_steps["model"]
    prep = rf_pipe.named_steps["preprocessor"]

    try:
        feature_names = prep.get_feature_names_out()
        importances = rf.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(15)

        fig = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            title="Top 15 Features - Random Forest"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(importance_df, use_container_width=True)
    except Exception as exc:
        st.warning(f"Feature importance could not be displayed: {exc}")

st.markdown("---")
st.caption("Academic project demonstration | Customer Churn Prediction using Machine Learning")
