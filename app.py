import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

@st.cache_resource
def load_artifacts():
    model          = joblib.load("model/best_model.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    all_columns    = joblib.load("model/all_columns_after_encoding.pkl")
    metadata       = joblib.load("model/metadata.pkl")
    scaler         = joblib.load("model/scaler.pkl") if metadata.get("requires_scaler") else None
    return model, scaler, feature_columns, all_columns, metadata

model, scaler, feature_columns, all_columns, metadata = load_artifacts()

st.title("📊 Customer Churn Prediction")
st.markdown("Aplikasi prediksi churn pelanggan berbasis machine learning.")
st.markdown("---")

with st.sidebar:
    st.header("ℹ️ Info Model")
    st.write(f"**Model:** {metadata['model_name']}")
    m = metadata["metrics"]
    st.metric("Accuracy",  f"{m['accuracy']:.4f}")
    st.metric("Precision", f"{m['precision']:.4f}")
    st.metric("Recall",    f"{m['recall']:.4f}")
    st.metric("F1-Score",  f"{m['f1']:.4f}")
    st.markdown("---")
    st.caption("UAS Bengkel Koding Data Science")

st.header("📝 Input Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    satisfaction_score        = st.slider("Satisfaction Score", 0.0, 10.0, 7.0)
    total_spent               = st.number_input("Total Spent ($)", 0.0, value=500.0)
    support_tickets           = st.number_input("Support Tickets", 0, value=1)
    delivery_delay_days       = st.number_input("Delivery Delay (hari)", 0, value=0)

with col2:
    nps_score                 = st.slider("NPS Score", -100, 100, 30)
    total_visits              = st.number_input("Total Visits", 0, value=20)
    last_3_month_purchase_freq = st.number_input("Pembelian 3 Bulan Terakhir", 0, value=3)
    marketing_spend_per_user  = st.number_input("Marketing Spend per User", 0.0, value=10.0)

st.markdown("---")

if st.button("🔮 Prediksi Churn", type="primary", use_container_width=True):

    # ✅ Hanya 8 fitur terpenting — sesuai variabel input di atas
    top8 = [
        'satisfaction_score',
        'total_spent',
        'support_tickets',
        'delivery_delay_days',
        'nps_score',
        'total_visits',
        'last_3_month_purchase_freq',
        'marketing_spend_per_user',
    ]

    user_data = {
        'satisfaction_score':         satisfaction_score,
        'total_spent':                total_spent,
        'support_tickets':            support_tickets,
        'delivery_delay_days':        delivery_delay_days,
        'nps_score':                  nps_score,
        'total_visits':               total_visits,
        'last_3_month_purchase_freq': last_3_month_purchase_freq,
        'marketing_spend_per_user':   marketing_spend_per_user,
    }

    # Buat DataFrame dengan SEMUA kolom dari training (isi 0 sebagai default)
    input_df = pd.DataFrame([{col: 0 for col in all_columns}])

    # Isi hanya 8 fitur yang diinput user
    for k, v in user_data.items():
        if k in input_df.columns:
            input_df[k] = v

    input_df = input_df[all_columns]

    # Scaling (jika model membutuhkan)
    if scaler is not None:
        scaled = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled, columns=all_columns)
    else:
        scaled_df = input_df

    # Pilih hanya fitur yang dipakai model saat training
    final_input = scaled_df[feature_columns]

    # Prediksi
    pred = model.predict(final_input)[0]
    try:
        prob = model.predict_proba(final_input)[0][1]
    except:
        prob = None

    st.markdown("## 📊 Hasil Prediksi")
    if pred == 1:
        st.error("⚠️ **Pelanggan diprediksi CHURN**")
        if prob is not None:
            st.write(f"Probabilitas churn: **{prob*100:.2f}%**")
        st.warning("💡 Rekomendasi: Lakukan retention campaign — diskon, hubungi customer service.")
    else:
        st.success("✅ **Pelanggan diprediksi TIDAK CHURN**")
        if prob is not None:
            st.write(f"Probabilitas churn: **{prob*100:.2f}%**")
        st.info("💡 Pelanggan stabil. Pertahankan kualitas layanan.")

    if prob is not None:
        st.progress(float(prob))

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | UAS Bengkel Koding 2025/2026")
