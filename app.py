import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import io
import os

# 1. CẤU HÌNH TRANG WEB STREAMLIT
st.set_page_config(
    page_title="Hệ thống phát hiện giao dịch bất thường trong KTNB",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS TÙY CHỈNH CHO GIAO DIỆN PREMIUM
st.markdown("""
    <style>
        .main-title { background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 800; margin-bottom: 0.2rem; }
        .kpi-container { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
        .kpi-card { flex: 1; background: linear-gradient(135deg, #151b2c 0%, #0b0f19 100%); border: 1px solid #2d3748; border-radius: 12px; padding: 1.25rem; text-align: center; }
        .kpi-value { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
        .kpi-label { font-size: 0.85rem; color: #9ca3af; text-transform: uppercase; }
        /* Style cho phần dữ liệu thô */
        .data-header { font-size: 1.8rem; font-weight: 700; margin-bottom: 1rem; }
        .metric-blue { color: #3b82f6; font-weight: 800; font-size: 2rem; }
    </style>
""", unsafe_allow_html=True)

# 3. HÀM XỬ LÝ (GIỮ NGUYÊN LOGIC)
@st.cache_data(show_spinner=False)
def load_data(file_path_or_buffer):
    df = pd.read_csv(file_path_or_buffer, parse_dates=["transaction_date"], dayfirst=False)
    return df

# (Các hàm khác như preprocess_and_train... giữ nguyên như cũ)
def preprocess_and_train(df, contamination, n_estimators, random_state):
    df_copy = df.copy()
    df_copy["gio_giao_dich"] = pd.to_datetime(df_copy["transaction_date"]).dt.hour
    df_copy["co nhan vien"] = df_copy["is_employee"].astype(int)
    x_features = df_copy[["amount", "gio_giao_dich", "co nhan vien"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x_features)
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    iso.fit(X_scaled)
    df_copy["anomaly_score"] = iso.decision_function(X_scaled)
    df_copy["is_anomaly"] = iso.predict(X_scaled) == -1
    return df_copy, iso, scaler

# 4. GIAO DIỆN CHÍNH
st.markdown('<div class="main-title">🛡️ HỆ THỐNG PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG</div>', unsafe_allow_html=True)

# LOAD DATA ĐỂ HIỂN THỊ THÔNG TIN
uploaded_file = st.sidebar.file_uploader("Tải lên tệp CSV", type=["csv"])
df = load_data(uploaded_file) if uploaded_file else pd.DataFrame()

# PHẦN HIỂN THỊ CẤU TRÚC DỮ LIỆU THÔ (ĐÃ CHỈNH SỬA)
if not df.empty:
    st.markdown('<div class="data-header">Phân tích cấu trúc dữ liệu thô</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Số lượng bản ghi (Rows)", f"{len(df):,}")
    c2.metric("Số lượng trường dữ liệu (Columns)", f"{len(df.columns)}")
    size_mb = uploaded_file.size / (1024 * 1024)
    c3.metric("Dung lượng tệp tin", f"{size_mb:.2f} MB")
    
    # Ép màu xanh dương và in đậm bằng CSS
    st.markdown("""
    <style>
        [data-testid="stMetricValue"] { color: #3b82f6 !important; font-weight: 800 !important; }
    </style>
    """, unsafe_allow_html=True)

# 5. CẤU HÌNH BIỂU ĐỒ (ĐÃ SỬA LỖI VALUEERROR)
if not df.empty:
    fig_hour = go.Figure()
    # ... (code thêm trace vào fig_hour giữ nguyên) ...
    
    # Cấu hình layout KHÔNG CÒN 'hovermode' để tránh lỗi
    fig_hour.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Giờ trong ngày'),
        yaxis=dict(title='Tổng số'),
        yaxis2=dict(title='Bất thường', overlaying='y', side='right')
    )
    st.plotly_chart(fig_hour, use_container_width=True)
