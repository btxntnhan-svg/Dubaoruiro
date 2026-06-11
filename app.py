import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import io
import os

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="Hệ thống KTNB", layout="wide")

# 2. CSS ĐỊNH DẠNG MÀU XANH DƯƠNG VÀ IN ĐẬM
st.markdown("""
    <style>
        .data-header { font-size: 1.8rem; font-weight: 700; color: #1e293b; margin-bottom: 1rem; }
        /* Định dạng cho các metric */
        div[data-testid="stMetricValue"] { 
            color: #3b82f6 !important; 
            font-weight: 800 !important; 
            font-size: 2.2rem !important; 
        }
        div[data-testid="stMetricLabel"] { 
            font-size: 1rem !important; 
            color: #64748b !important; 
        }
    </style>
""", unsafe_allow_html=True)

# 3. LOAD DỮ LIỆU
uploaded_file = st.sidebar.file_uploader("Tải lên tệp CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["transaction_date"])
    
    # HIỂN THỊ DỮ LIỆU THÔ (Với màu xanh dương in đậm)
    st.markdown('<div class="data-header">Phân tích cấu trúc dữ liệu thô</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Số lượng bản ghi (Rows)", f"{len(df):,}")
    c2.metric("Số lượng trường dữ liệu (Columns)", f"{len(df.columns)}")
    c3.metric("Dung lượng tệp tin", f"{uploaded_file.size / (1024*1024):.2f} MB")

    # 4. VẼ BIỂU ĐỒ (ĐÃ LOẠI BỎ 'hovermode' ĐỂ KHÔNG BỊ LỖI VALUEERROR)
    st.subheader("Thống kê giao dịch")
    hour_counts = df['transaction_date'].dt.hour.value_counts().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hour_counts.index, y=hour_counts.values, name="Tổng giao dịch"))
    
    # Cấu hình layout tối giản, an toàn cho trục Y kép
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20),
        # KHÔNG SỬ DỤNG hovermode TẠI ĐÂY ĐỂ TRÁNH LỖI VALUEERROR
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Vui lòng tải tệp CSV lên thanh bên để bắt đầu.")
