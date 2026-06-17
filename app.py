import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Surgery Recovery Predictor", layout="centered")

# ================= LOAD DATA =================
df = pd.read_excel("surgery recovery dataset.xlsx")
model = joblib.load("model.pkl")

# ================= THEME TOGGLE =================
theme = st.sidebar.toggle("🌗 Light / Dark Mode", value=False)

# ================= VISIBILITY FIX (ONLY COLORS) =================
if theme:
    text_color = "#C09494"
    bg = "#020617"
    accent = "#6017e7"
    input_bg = "#838488"
else:
    text_color = "#1a1d24"
    bg = "#DEF59F"
    accent = "#c9def3"
    input_bg = "#f8b0b0"

st.markdown(f"""
<style>
/* GLOBAL TEXT */
html, body, * {{
    color: {text_color} !important;
    -webkit-text-fill-color: {text_color} !important;
    caret-color: {text_color} !important;
}}

/* APP BACKGROUND */
.stApp {{
    background-color: {bg} !important;
}}

/* SIDEBAR */
[data-testid="stSidebar"] {{
    background-color: {accent} !important;
}}

[data-testid="stSidebar"] * {{
    color: white !important;
    -webkit-text-fill-color: white !important;
}}

/* INPUT FIELDS */
input, textarea {{
    color: {text_color} !important;
    background-color: {input_bg} !important;
    -webkit-text-fill-color: {text_color} !important;
}}

/* SELECTBOX VALUE */
div[data-baseweb="select"] > div {{
    color: {text_color} !important;
    background-color: {input_bg} !important;
}}

/* SELECTBOX DROPDOWN */
div[role="listbox"] div {{
    color: {text_color} !important;
    background-color: {input_bg} !important;
}}

/* SLIDER NUMBERS */
span {{
    color: {text_color} !important;
    -webkit-text-fill-color: {text_color} !important;
}}

/* FIX + / - BUTTONS */
button svg, button svg path {{
    fill: {text_color} !important;
    stroke: {text_color} !important;
}}

/* MAIN BUTTON */
.stButton > button {{
    background: linear-gradient(90deg, {accent}, #1e40af);
    font-weight: 700;
    font-size: 18px;
    border-radius: 12px;
    color: white !important;
}}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🏥 Predicting Recovery Time After Surgery")
st.write("Machine Learning based recovery time prediction system")

# ================= SIDEBAR =================
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 95)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
bmi = st.sidebar.number_input("BMI", 15.0, 50.0, 25.0)

surgery = st.sidebar.selectbox(
    "Type of Surgery",
    ["Cardiac","Orthopedic","General","Neuro","Urology","Gynecology","ENT",
     "Plastic","Thoracic","Bariatric","Ophthalmology","Colorectal","Vascular",
     "Spine","Oncology","Trauma"]
)

conditions = st.sidebar.slider("Pre-existing Conditions", 0, 6)
blood_loss = st.sidebar.number_input("Blood Loss (ml)", 0, 4000, 500)
duration = st.sidebar.number_input("Surgery Duration (minutes)", 30, 720, 120)
icu = st.sidebar.slider("ICU Stay (days)", 0, 14)
complications = st.sidebar.selectbox("Complications", ["No", "Yes"])
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
exercise = st.sidebar.selectbox("Exercise Level", ["Low", "Moderate", "High"])
hemoglobin = st.sidebar.number_input("Hemoglobin", 8.0, 18.0, 13.5)
glucose = st.sidebar.number_input("Blood Glucose", 70, 250, 120)
alcohol = st.sidebar.selectbox("Alcohol Use", ["No", "Yes"])
platelets = st.sidebar.number_input("Platelets", 120, 450, 250)
creatinine = st.sidebar.number_input("Creatinine", 0.4, 3.0, 1.0)
wbc = st.sidebar.number_input("WBC Count", 3.0, 15.0, 7.5)

# ================= ENCODING =================
gender = 1 if gender == "Male" else 0
complications = 1 if complications == "Yes" else 0
smoker = 1 if smoker == "Yes" else 0
exercise = {"Low":0,"Moderate":1,"High":2}[exercise]
alcohol = 1 if alcohol == "Yes" else 0

surgery_map = {name:i for i,name in enumerate([
    "Cardiac","Orthopedic","General","Neuro","Urology","Gynecology","ENT",
    "Plastic","Thoracic","Bariatric","Ophthalmology","Colorectal","Vascular",
    "Spine","Oncology","Trauma"
])}
surgery = surgery_map[surgery]

# ================= PREDICTION =================
prediction = None
if st.button("🔮 Predict Recovery Time"):
    input_data = np.array([[age, gender, bmi, surgery, conditions,
                             blood_loss, duration, complications,
                             icu, smoker, exercise, hemoglobin, glucose,
                             platelets, creatinine, alcohol, wbc]])
    prediction = model.predict(input_data)[0]
    st.success(f"🛌 Estimated Recovery Time: **{prediction:.1f} days**")

# ===== RECOVERY RISK CLASSIFICATION =====
if prediction is not None:
    risk_label = None

    if prediction <= 5:
        risk_label = "🟢 Low Risk"
        risk_color = "#16a34a"
        risk_msg = "Fast recovery expected with minimal complications."

    elif prediction <= 12:
        risk_label = "🟡 Medium Risk"
        risk_color = "#ca8a04"
        risk_msg = "Moderate recovery time. Follow medical advice carefully."

    else:
        risk_label = "🔴 High Risk"
        risk_color = "#dc2626"
        risk_msg = "Long recovery expected. Close monitoring recommended."

    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {risk_color};
            background-color: #f8fafc;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        ">
            <h3 style="color:{risk_color}; margin-bottom:5px;">
                🚦 Recovery Risk Level: {risk_label}
            </h3>
            <p style="font-size:16px;">{risk_msg}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
# ===== DOCTOR ADVICE BASED ON RISK =====
    if risk_label == "🟢 Low Risk":
        advice = "Maintain light physical activity, follow medication schedule, and attend follow-ups."
    elif risk_label == "🟡 Medium Risk":
        advice = "Monitor vitals regularly, avoid strenuous activity, and follow diet guidelines."
    else:
        advice = "Strict monitoring required. Immediate medical attention if symptoms worsen."

    st.info(f"👨‍⚕️ Doctor's Advice: {advice}")

# ================= GRAPHS =================
st.subheader("📊 Recovery Insights")
Recovery_Days= [c for c in df.columns if "recovery" in c.lower()][0]
Type_of_Surgery = [c for c in df.columns if "surgery" in c.lower()][0]

st.bar_chart(df[Type_of_Surgery].value_counts())
st.line_chart(df.groupby("Age")[Recovery_Days].mean())

# ===== RISK DISTRIBUTION PIE CHART =====
st.subheader("🚦 Recovery Risk Distribution")

risk_counts = {
    "Low Risk": (df[Recovery_Days] <= 5).sum(),
    "Medium Risk": ((df[Recovery_Days] > 5) & (df[Recovery_Days] <= 12)).sum(),
    "High Risk": (df[Recovery_Days] > 12).sum()
}

fig, ax = plt.subplots()
ax.pie(
    risk_counts.values(),
    labels=risk_counts.keys(),
    autopct="%1.1f%%",
    startangle=90
)
ax.axis("equal")
st.pyplot(fig)


# ================= PATIENT REPORT PDF =================
if prediction:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFillColor(HexColor("#1976d2"))
    c.rect(0, 730, 612, 80, fill=1)

    c.setFillColor(HexColor("#f1a5a5"))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, 760, "🏥 Surgery Recovery Report")

    c.setFillColor(HexColor("#000000"))
    c.setFont("Helvetica", 14)
    y = 680

    details = {
        "Age of the Patient": age,
        "BMI": bmi,
        "Surgery Type": list(surgery_map.keys())[list(surgery_map.values()).index(surgery)],
        "Blood Loss": blood_loss,
        "Surgery Duration": duration,
        "ICU Stay": icu,
        "Glucose": glucose,
        "Hemoglobin": hemoglobin,
        "Platelets": platelets,
        "WBC": wbc,
        "Estimated Recovery": f"{prediction:.1f} days",
        "Recovery Risk Level": risk_label
    }

    for k, v in details.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 28

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y-10, "Doctor's Advice:")

    c.setFont("Helvetica", 12)
    c.drawString(50, y-35, advice)

    y -= 60

    c.setStrokeColor(HexColor("#1976d2"))
    c.line(50, y, 560, y)

    c.setFont("Helvetica-Oblique", 12)
    c.drawString(50, y-40, " *END OF REPORT* ")

    c.showPage()
    c.save()
    buffer.seek(0)

    st.download_button(
        "📄 Download Patient Report",
        buffer,
        file_name="patient_report.pdf",
        mime="application/pdf"
    )

# ================= FOOTER =================
st.markdown("---")
st.caption(" Report ")
