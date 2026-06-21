"""
Streamlit front-end for the Visit with Us Wellness Tourism Package predictor.
Loads the trained sklearn Pipeline from Hugging Face Model Hub and serves
interactive predictions.
"""
import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "ssingh94/tourism-model")
HF_TOKEN      = os.environ.get("HF_TOKEN", None)


@st.cache_resource(show_spinner="Loading model from Hugging Face…")
def load_model():
    path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.pkl", token=HF_TOKEN)
    return joblib.load(path)


model = load_model()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visit with Us — Package Predictor",
    page_icon="✈️",
    layout="wide",
)
st.title("✈️ Wellness Tourism Package — Purchase Predictor")
st.markdown(
    "Enter customer details below and click **Predict** to determine the likelihood "
    "of purchasing the Wellness Tourism Package."
)
st.divider()

# ─── Input Form ───────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Customer Profile")
        age            = st.slider("Age", 18, 65, 35)
        gender         = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        occupation     = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
        designation    = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, max_value=100000,
                                         value=20000, step=500)

    with col2:
        st.subheader("Travel Preferences")
        city_tier       = st.selectbox("City Tier", [1, 2, 3])
        num_persons     = st.slider("Number of Persons Visiting", 1, 5, 2)
        num_children    = st.slider("Children Below Age 5", 0, 3, 0)
        preferred_star  = st.selectbox("Preferred Property Stars", [3, 4, 5])
        num_trips       = st.slider("Annual Number of Trips", 1, 10, 3)
        passport        = st.radio("Has Passport?",  [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
        own_car         = st.radio("Owns a Car?",    [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)

    with col3:
        st.subheader("Sales Interaction")
        type_of_contact  = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
        product_pitched  = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
        pitch_score      = st.slider("Pitch Satisfaction Score (1–5)", 1, 5, 3)
        num_followups    = st.slider("Number of Follow-ups", 1, 6, 3)
        duration_pitch   = st.slider("Duration of Pitch (minutes)", 5, 60, 15)

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

# ─── Prediction ───────────────────────────────────────────────────────────────
if submitted:
    input_df = pd.DataFrame([{
        "Age":                      age,
        "TypeofContact":            type_of_contact,
        "CityTier":                 city_tier,
        "DurationOfPitch":          duration_pitch,
        "Occupation":               occupation,
        "Gender":                   gender,
        "NumberOfPersonVisiting":   num_persons,
        "NumberOfFollowups":        num_followups,
        "ProductPitched":           product_pitched,
        "PreferredPropertyStar":    preferred_star,
        "MaritalStatus":            marital_status,
        "NumberOfTrips":            num_trips,
        "Passport":                 passport,
        "PitchSatisfactionScore":   pitch_score,
        "OwnCar":                   own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation":              designation,
        "MonthlyIncome":            monthly_income,
    }])

    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    col_r1, col_r2, col_r3 = st.columns([2, 1, 1])

    with col_r1:
        if prediction == 1:
            st.success(f"### ✅ WILL PURCHASE the Wellness Tourism Package")
        else:
            st.error(f"### ❌ WILL NOT PURCHASE the Wellness Tourism Package")

    with col_r2:
        st.metric("Purchase Probability", f"{probability:.1%}")

    with col_r3:
        st.metric("Non-Purchase Probability", f"{1 - probability:.1%}")
