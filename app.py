import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/best_model.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    all_columns = joblib.load("model/all_columns_after_encoding.pkl")
    metadata = joblib.load("model/metadata.pkl")
    scaler = joblib.load("model/scaler.pkl") if metadata.get("requires_scaler") else None
    return model, scaler, feature_columns, all_columns, metadata

model, scaler, feature_columns, all_columns, metadata = load_artifacts()

st.title("📊 Customer Churn Prediction")
st.markdown("Aplikasi prediksi churn pelanggan berbasis machine learning.")
st.markdown("---")

with st.sidebar:
    st.header("ℹ️ Info Model")
    st.write(f"**Model:** {metadata["model_name"]}")
    m = metadata["metrics"]
    st.metric("Accuracy",  f"{m["accuracy"]:.4f}")
    st.metric("Precision", f"{m["precision"]:.4f}")
    st.metric("Recall",    f"{m["recall"]:.4f}")
    st.metric("F1-Score",  f"{m["f1"]:.4f}")
    st.markdown("---")
    st.caption("UAS Bengkel Koding Data Science")

st.header("📝 Input Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demografi")
    age = st.number_input("Usia", 15, 100, 35)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    country = st.selectbox("Negara", ["USA", "UK", "Indonesia", "India", "Germany", "Other"])
    is_premium = st.selectbox("Premium User?", [0, 1])

with col2:
    st.subheader("Aktivitas")
    total_visits = st.number_input("Total Visits", 0, value=20)
    avg_session_time = st.number_input("Avg Session Time", 0.0, value=15.0)
    pages_per_session = st.number_input("Pages per Session", 0.0, value=5.0)
    email_open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.5)
    email_click_rate = st.slider("Email Click Rate", 0.0, 1.0, 0.2)

with col3:
    st.subheader("Transaksi")
    total_spent = st.number_input("Total Spent ($)", 0.0, value=500.0)
    avg_order_value = st.number_input("Avg Order Value ($)", 0.0, value=50.0)
    discount_used = st.selectbox("Discount Used?", [0, 1])
    support_tickets = st.number_input("Support Tickets", 0, value=1)
    refund_requested = st.selectbox("Refund Requested?", [0, 1])
    delivery_delay = st.number_input("Delivery Delay (hari)", 0, value=0)
    satisfaction = st.slider("Satisfaction Score", 0.0, 10.0, 7.0)
    nps_score = st.slider("NPS Score", -100, 100, 30)
    marketing_spend = st.number_input("Marketing Spend", 0.0, value=10.0)
    lifetime_value = st.number_input("Lifetime Value", 0.0, value=1000.0)
    last_3_freq = st.number_input("Pembelian 3 Bulan Terakhir", 0, value=3)
    tenure_days = st.number_input("Tenure (hari)", 0, value=365)
    days_since_last = st.number_input("Hari Sejak Pembelian Terakhir", 0, value=30)

st.subheader("Layanan & Pembayaran")
c1, c2, c3, c4 = st.columns(4)
with c1:
    acquisition = st.selectbox("Acquisition Channel", ["Email", "Ads", "Organic", "Referral", "Social"])
with c2:
    device = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
with c3:
    subscription = st.selectbox("Subscription Type", ["Basic", "Premium", "Enterprise"])
with c4:
    payment = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Bank Transfer", "Crypto"])

st.markdown("---")
if st.button("🔮 Prediksi Churn", type="primary", use_container_width=True):
    user_data = {
        "age": age, "is_premium_user": is_premium,
        "total_visits": total_visits, "avg_session_time": avg_session_time,
        "pages_per_session": pages_per_session,
        "email_open_rate": email_open_rate, "email_click_rate": email_click_rate,
        "total_spent": total_spent, "avg_order_value": avg_order_value,
        "discount_used": discount_used, "support_tickets": support_tickets,
        "refund_requested": refund_requested, "delivery_delay_days": delivery_delay,
        "satisfaction_score": satisfaction, "nps_score": nps_score,
        "marketing_spend_per_user": marketing_spend, "lifetime_value": lifetime_value,
        "last_3_month_purchase_freq": last_3_freq,
        "days_since_last_purchase": days_since_last, "tenure_days": tenure_days,
        f"gender_{gender}": 1, f"country_{country}": 1,
        f"acquisition_channel_{acquisition}": 1, f"device_type_{device}": 1,
        f"subscription_type_{subscription}": 1, f"payment_method_{payment}": 1,
    }

    # Buat dataframe dengan SEMUA kolom dari training
    input_df = pd.DataFrame([{col: 0 for col in all_columns}])
    for k, v in user_data.items():
        if k in input_df.columns:
            input_df[k] = v
    input_df = input_df[all_columns]

    # Scaling
    if scaler is not None:
        scaled = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled, columns=all_columns)
    else:
        scaled_df = input_df

    # Pilih hanya feature yang dipakai model
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
