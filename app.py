import streamlit as st
import pandas as pd
import pickle
from fpdf import FPDF
import matplotlib.pyplot as plt
import difflib

st.set_page_config(page_title="🧠 AI-Powered Medicine & Disease Prediction System", page_icon="🧬", layout="wide")

# --- LOAD DATA AND MODEL ---
clf = pickle.load(open("model.pkl", "rb"))
with open("symptom_columns.pkl", "rb") as f:
    symptoms_list = pickle.load(f)
medicine_df = pd.read_csv("medicine_disease.csv")  # columns: 'disease', 'drug'
try:
    disease_df = pd.read_csv("diseases.csv")       # columns: 'name', 'description', 'causes', 'prevention'
except FileNotFoundError:
    disease_df = pd.DataFrame()

# --- HEADER AND SIDEBAR ---
st.markdown(
    """
    <h1 style='text-align: center;'>🧬 AI-Powered Medicine & Disease Prediction System</h1>
    <h4 style='text-align: center; color: gray;'>Futuristic, Smart Disease Prediction & Medicine Recommendation System 🌡️</h4>
    <hr>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("💡 Health Tips for You")
st.sidebar.info("""
- Drink plenty of water 💧  
- Get enough sleep 😴  
- Exercise regularly 🏃‍♂️  
- Eat more fruits & vegetables 🥗  
- Avoid excessive sugar & caffeine 🚫  
- Wash hands frequently 🧼  
- Take breaks! 
"""
)
st.sidebar.markdown("---")
st.sidebar.write("Disclaimer: For education only. See a doctor for medical treatment.")

# --- SYMPTOM INPUT ---
st.subheader("🤒 Enter Your Symptoms")
user_symptoms = st.multiselect("Select symptoms (choose multiple):", symptoms_list)

def get_disease_info(pred):
    info = {}
    if not disease_df.empty:
        row = disease_df[disease_df['name'].str.lower() == pred.lower()]
        if not row.empty:
            info = {
                'Description': row.iloc[0].get('description', '—'),
                'Causes': row.iloc[0].get('causes', '—'),
                'Prevention': row.iloc[0].get('prevention', '—')
            }
    return info

# --- PREDICTION & OUTPUT ---
if st.button("🔍 Predict Disease"):
    if not user_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = [1 if s in user_symptoms else 0 for s in symptoms_list]
        pred = clf.predict([input_vector])[0]
        st.success(f"🩺 **Predicted Disease:** {pred}")

        # --- Fuzzy Matching for Disease Name ---
        csv_diseases = medicine_df['disease'].str.lower().unique()
        best_match = difflib.get_close_matches(pred.lower(), csv_diseases, n=1, cutoff=0.8)
        if best_match:
            med_rows = medicine_df[medicine_df['disease'].str.lower() == best_match[0]]
        else:
            med_rows = pd.DataFrame()

        info = get_disease_info(pred)
        with st.expander("🩺 Disease Info & Prevention"):
            if info:
                st.write(f"**Description:** {info['Description']}")
                st.write(f"**Causes:** {info['Causes']}")
                st.write(f"**Prevention:** {info['Prevention']}")
            else:
                st.write("No extra disease info available for this prediction.")

        # --- Medicine Recommendations Section: Top 8 split into important/optional ---
        if not med_rows.empty:
            st.subheader("💊 Important Medicines")
            important_meds = med_rows.head(4)
            for _, row in important_meds.iterrows():
                st.write(f"- 🟢 **{row['drug']}**")

            optional_meds = med_rows.iloc[4:8]
            if not optional_meds.empty:
                st.markdown("#### Other Optional Medicines (can use if needed):")
                for _, row in optional_meds.iterrows():
                    st.write(f"- {row['drug']}")
        else:
            st.info("No mapped medicines found for this disease.")

        # Chart Visualization
        st.subheader("🩹 Selected Symptoms Overview")
        if user_symptoms:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.bar(user_symptoms, [len(s) for s in user_symptoms], color="#33c3f0")
            ax.set_xlabel("Symptoms")
            ax.set_ylabel("Demo Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # --- PDF Report Generation ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=13)
        pdf.cell(0, 10, "AI Health Prediction Report", ln=True, align='C')
        pdf.cell(0, 8, f"Disease predicted: {pred}", ln=True)
        pdf.cell(0, 8, "Symptoms provided: " + ", ".join(user_symptoms), ln=True)
        if not med_rows.empty:
            pdf.cell(0, 8, "Important Medicine(s):", ln=True)
            for _, row in important_meds.iterrows():
                pdf.cell(0, 8, f"{row['drug']}", ln=True)
            if not optional_meds.empty:
                pdf.cell(0, 8, "Other Optional Medicine(s):", ln=True)
                for _, row in optional_meds.iterrows():
                    pdf.cell(0, 8, f"{row['drug']}", ln=True)
        pdf.output("health_report.pdf")
        with open("health_report.pdf", "rb") as f:
            st.download_button("📥 Download Health Report (PDF)", f, file_name="health_report.pdf")

        st.info(
            "🛸 **AI Tip:** This system shows the most important medicines first and additional optional ones for your safety. Always consult a doctor."
        )

st.markdown("""<hr><center>
⚡ <b>AI Health Advisor 2025</b> | Advanced ML Demo — Ready to expand 🚀
</center>""", unsafe_allow_html=True)
