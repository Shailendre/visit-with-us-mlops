"""
Visit with Us — Wellness Tourism Package Purchase Predictor
Left panel: all inputs | Right panel: prediction result
"""
import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "ssingh94/tourism-model")
HF_TOKEN      = os.environ.get("HF_TOKEN", None)

st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model.pkl", token=HF_TOKEN)
    return joblib.load(path)

# ─── Title ────────────────────────────────────────────────────────────────────
st.title("✈️ Wellness Tourism Package — Purchase Predictor")
st.caption("Fill in customer details on the left and click **Predict**.")
st.divider()

# ─── Two-panel layout ─────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("📋 Customer Details")
    with st.form("prediction_form"):

        a, b = st.columns(2)

        with a:
            age             = st.slider("Age", 18, 65, 35)
            gender          = st.selectbox("Gender", ["Female", "Male"])
            marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            occupation      = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
            designation     = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            monthly_income  = st.number_input("Monthly Income (₹)", 1000, 100000, 20000, step=500)
            city_tier       = st.selectbox("City Tier", [1, 2, 3])
            type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
            product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])

        with b:
            num_persons    = st.slider("Persons Visiting", 1, 5, 2)
            num_children   = st.slider("Children Below Age 5", 0, 3, 0)
            preferred_star = st.selectbox("Preferred Property Stars", [3, 4, 5])
            num_trips      = st.slider("Annual Trips", 1, 10, 3)
            passport       = st.radio("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
            own_car        = st.radio("Owns a Car?",   [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
            pitch_score    = st.slider("Pitch Satisfaction (1–5)", 1, 5, 3)
            num_followups  = st.slider("Number of Follow-ups", 1, 6, 3)
            duration_pitch = st.slider("Duration of Pitch (mins)", 5, 60, 15)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

# ─── Right panel: result ──────────────────────────────────────────────────────
with right:
    st.subheader("📊 Prediction Result")

    if submitted:
        try:
            model = load_model()
        except Exception as e:
            st.error(f"**Model failed to load:** {e}")
            st.stop()

        input_df = pd.DataFrame([{
            "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
            "DurationOfPitch": duration_pitch, "Occupation": occupation, "Gender": gender,
            "NumberOfPersonVisiting": num_persons, "NumberOfFollowups": num_followups,
            "ProductPitched": product_pitched, "PreferredPropertyStar": preferred_star,
            "MaritalStatus": marital_status, "NumberOfTrips": num_trips, "Passport": passport,
            "PitchSatisfactionScore": pitch_score, "OwnCar": own_car,
            "NumberOfChildrenVisiting": num_children, "Designation": designation,
            "MonthlyIncome": monthly_income,
        }])

        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.session_state["pred"]  = int(prediction)
        st.session_state["prob"]  = float(probability)
        st.session_state["ready"] = True

    if st.session_state.get("ready"):
        pred = st.session_state["pred"]
        prob = st.session_state["prob"]

        st.markdown("---")
        if pred == 1:
            st.success("## ✅ WILL PURCHASE")
            st.markdown("This customer is **likely to buy** the Wellness Tourism Package.")
        else:
            st.error("## ❌ WILL NOT PURCHASE")
            st.markdown("This customer is **unlikely to buy** the Wellness Tourism Package.")

        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Purchase Probability",     f"{prob:.1%}")
        m2.metric("Non-Purchase Probability", f"{1 - prob:.1%}")

        st.markdown("---")
        st.progress(prob)
        st.caption(f"Confidence: {max(prob, 1-prob):.1%}")

    else:
        st.info("👈 Fill in customer details and click **Predict** to see the result here.")
