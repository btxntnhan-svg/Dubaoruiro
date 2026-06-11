import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import io

# ==========================================
# 0. INITIAL CONFIGURATION
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ Thống Phát Hiện Giao Dịch Gian Lận",
    page_icon="🛡️"
)

# ==========================================
# 1. SHARED FUNCTIONS & CACHING
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Đọc dữ liệu mẫu từ bytes để tối ưu hóa bộ nhớ và hashable"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            st.error("Định dạng file không được hỗ trợ! Vui lòng tải file .csv hoặc .xlsx")
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None

def train_model_pipeline(df, n_estimators, max_depth, random_state):
    """Pipeline huấn luyện mô hình và tính toán kết quả kiểm định"""
    # Định nghĩa tập biến đầu vào và biến mục tiêu dựa theo notebook
    features = [f'X_{i}' for i in range(1, 15)]
    target = 'default'
    
    # Kiểm tra tính toàn vẹn của dữ liệu
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        st.error(f"Dữ liệu thiếu các cột bắt buộc: {missing_cols}")
        return None

    X = df[features]
    y = df[target]

    # Phân chia tập dữ liệu huấn luyện/kiểm tra 80/20 giống quy trình chuẩn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    
    # Khởi tạo và huấn luyện mô hình RandomForest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth != 0 else None,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Dự đoán trên tập kiểm tra để đánh giá hiệu năng
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Tổng hợp các chỉ số kiểm định
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return model, metrics, features

# ==========================================
# 2. SIDEBAR - CONFIGURATION ZONE
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải tệp dữ liệu lên hệ thống
    uploaded_file = st.file_uploader(
        "Tải lên tập dữ liệu huấn luyện", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu CSV hoặc Excel chứa các cột từ X_1 đến X_14 và cột mục tiêu 'default'."
    )
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: RandomForestClassifier")
    
    # Điều chỉnh siêu tham số mô hình phù hợp với Notebook
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, max_value=300, value=100, step=10,
        help="Số lượng cây quyết định trong rừng cây."
    )
    
    max_depth_choice = st.selectbox(
        "Độ sâu tối đa (max_depth)",
        options=["Không giới hạn", "5", "10", "15", "20", "25"],
        index=0,
        help="Độ sâu tối đa của mỗi cây quyết định."
    )
    max_depth = 0 if max_depth_choice == "Không giới hạn" else int(max_depth_choice)
    
    with st.expander("⚙️ Tham số nâng cao"):
        random_state = st.number_input(
            "Trạng thái ngẫu nhiên (random_state)",
            min_value=0, max_value=9999, value=42, step=1,
            help="Đảm bảo khả năng tái lập kết quả huấn luyện giữa các lần chạy."
        )

    st.divider()
    # Nút bấm hành động duy nhất để kích hoạt quy trình huấn luyện
    btn_train = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==========================================
# 3. HEADER ZONE & DATA CHECK
# ==========================================
st.title("🛡️ Hệ Thống Dự Báo & Phát Hiện Giao Dịch Gian Lận")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu rủi ro tài chính, tự động phát hiện các hành vi giao dịch bất thường dựa trên mô hình học máy RandomForest.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tệp dữ liệu (.csv hoặc .xlsx) từ thanh Sidebar bên trái để bắt đầu khai thác.")
    st.stop()

# Đọc dữ liệu sau khi đảm bảo file đã được chọn
file_bytes = uploaded_file.read()
df = load_data(file_bytes, uploaded_file.name)

if df is None:
    st.stop()

st.caption(f"📁 Đang sử dụng tệp dữ liệu: `{uploaded_file.name}` | Quy mô: **{df.shape[0]}** dòng, **{df.shape[1]}** cột")
st.divider()

# Xử lý sự kiện nhấn nút huấn luyện
if btn_train:
    with st.spinner("Hệ thống đang tiến hành huấn luyện và phân tích dữ liệu..."):
        res = train_model_pipeline(df, n_estimators, max_depth, random_state)
        if res is not None:
            st.session_state['model'] = res[0]
            st.session_state['metrics'] = res[1]
            st.session_state['features'] = res[2]
            st.success("🎉 Huấn luyện mô hình thành công! Kết quả đã được cập nhật tại các Tab bên dưới.")

# ==========================================
# 4. MAIN CONTENT - TABS INTERFACE
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả kiểm định mô hình", 
    "🔮 Triển khai sử dụng mô hình"
])

# ------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# ------------------------------------------
with tab1:
    st.subheader("Phân tích cấu trúc dữ liệu thô")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Số lượng bản ghi (Rows)", f"{df.shape[0]:,}")
    col2.metric("Số lượng trường dữ liệu (Columns)", f"{df.shape[1]}")
    file_size_mb = len(file_bytes) / (1024 * 1024)
    col3.metric("Dung lượng tệp tin", f"{file_size_mb:.2f} MB")
    
    st.write("#### 🕵️ 5 dòng dữ liệu đầu tiên")
    st.dataframe(df.head(5), use_container_width=True)
    
    st.write("#### 📐 Thống kê mô tả các biến đặc trưng (X_1 đến X_14)")
    features_list = [f'X_{i}' for i in range(1, 15) if f'X_{i}' in df.columns]
    if features_list:
        st.dataframe(df[features_list].describe().T, use_container_width=True)
    else:
        st.warning("Không tìm thấy các biến đặc trưng chuẩn (X_1 đến X_14) trong tập dữ liệu.")

# ------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------
with tab2:
    st.subheader("Biểu đồ phân phối cấu trúc biến")
    
    # Ưu tiên hiển thị biến mục tiêu trước tiên
    target_col = 'default'
    if target_col in df.columns:
        st.write("#### 🎯 Phân phối của biến mục tiêu (Gian lận vs Bình thường)")
        class_counts = df[target_col].value_counts().reset_index()
        class_counts.columns = ['Trạng thái', 'Số lượng']
        class_counts['Trạng thái'] = class_counts['Trạng thái'].map({0: '0 - Bình thường', 1: '1 - Gian lận/Rủi ro'})
        
        fig_target = px.bar(
            class_counts, x='Trạng thái', y='Số lượng',
            color='Trạng thái', color_discrete_sequence=px.colors.qualitative.Set2,
            text_auto=True, height=350
        )
        st.plotly_chart(fig_target, use_container_width=True)
    
    st.write("#### 📊 Trực quan chi tiết hệ thống biến số")
    selected_features = st.multiselect(
        "Chọn các biến số để hiển thị biểu đồ phân phối (Tối đa nên chọn 4 biến để tối ưu giao diện):",
        options=features_list,
        default=features_list[:4] if len(features_list) >= 4 else features_list
    )
    
    if selected_features:
        # Tạo lưới phân bố 2x2 cho biểu đồ hình học
        cols = st.columns(2)
        for idx, feat in enumerate(selected_features):
            current_col = cols[idx % 2]
            with current_col:
                fig_hist = px.histogram(
                    df, x=feat, color=target_col if target_col in df.columns else None,
                    marginal="box", barmode="overlay",
                    title=f"Phân phối tần suất biến {feat}",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    height=300
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Vui lòng lựa chọn ít nhất một biến để hiển thị đồ thị trực quan.")

# ------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------
with tab3:
    st.subheader("Đánh giá độ chính xác thuật toán AI")
    
    if 'model' not in st.session_state:
        st.info("ℹ️ Hiện chưa có dữ liệu mô hình. Vui lòng thiết lập cấu hình và bấm nút **[🚀 Huấn luyện mô hình]** tại thanh Sidebar.")
    else:
        metrics = st.session_state['metrics']
        
        # Hiển thị các chỉ số đo lường cốt lõi dạng thẻ (Metrics)
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Độ chính xác tổng thể (Accuracy)", f"{metrics['accuracy']:.4f}")
        m_col2.metric("Độ chính xác dự báo (Precision)", f"{metrics['precision']:.4f}")
        m_col3.metric("Tỷ lệ bắt sót (Recall / Sensitivity)", f"{metrics['recall']:.4f}")
        m_col4.metric("Chỉ số cân bằng F1-Score", f"{metrics['f1']:.4f}")
        
        st.divider()
        
        grid_col1, grid_col2 = st.columns(2)
        
        with grid_col1:
            st.write("#### 🧮 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = metrics['confusion_matrix']
            x_labels = ['Dự báo Thường (0)', 'Dự báo Gian lận (1)']
            y_labels = ['Thực tế Thường (0)', 'Thực tế Gian lận (1)']
            
            fig_cm = ff.create_annotated_heatmap(
                z=cm, x=x_labels, y=y_labels, 
                colorscale='Blues', showscale=True
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with grid_col2:
            st.write("#### 🌲 Độ quan trọng của các biến (Feature Importances)")
            trained_model = st.session_state['model']
            importances = trained_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Biến số': st.session_state['features'],
                'Mức độ đóng góp (%)': importances * 100
            }).sort_values(by='Mức độ đóng góp (%)', ascending=True)
            
            fig_imp = px.bar(
                feat_imp_df, x='Mức độ đóng góp (%)', y='Biến số',
                orientation='h', color='Mức độ đóng góp (%)',
                color_continuous_scale='Viridis', height=380
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH (DỰ BÁO THỰC TẾ)
# ------------------------------------------
with tab4:
    st.subheader("Khai thác mô hình dự báo trực tuyến")
    
    if 'model' not in st.session_state:
        st.info("ℹ️ Hiện chưa có dữ liệu mô hình phục vụ dự báo. Vui lòng bấm nút **[🚀 Huấn luyện mô hình]** trước.")
    else:
        trained_model = st.session_state['model']
        features = st.session_state['features']
        
        mode = st.radio(
            "Phương thức nạp dữ liệu cần dự báo:",
            options=["Nhập chỉ số đơn lẻ trực tiếp trên form", "Tải tệp danh sách khách hàng hàng loạt (Excel/CSV)"],
            horizontal=True
        )
        
        if mode == "Nhập chỉ số đơn lẻ trực tiếp trên form":
            st.write("#### 📝 Cấu hình chỉ số giao dịch tài chính cá nhân")
            
            # Tính toán phân phối nền để đặt giá trị mặc định cho form
            default_inputs = {}
            with st.form("single_prediction_form"):
                form_cols = st.columns(3)
                for idx, feat in enumerate(features):
                    col_target = form_cols[idx % 3]
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    mean_val = float(df[feat].median()) # Dùng median làm mặc định bền vững
                    
                    default_inputs[feat] = col_target.number_input(
                        f"Giá trị biến {feat}",
                        min_value=min_val - abs(min_val)*0.5,
                        max_value=max_val + abs(max_val)*0.5,
                        value=mean_val,
                        format="%.4f",
                        help=f"Khoảng giá trị thực tế trong tập mẫu: [{min_val:.2f} đến {max_val:.2f}]"
                    )
                
                submit_pred = st.form_submit_button("🔍 Tiến hành phân tích rủi ro", type="primary", use_container_width=True)
                
            if submit_pred:
                # Chuyển đổi định dạng dữ liệu vào DataFrame để dự đoán
                input_df = pd.DataFrame([default_inputs])
                prediction = trained_model.predict(input_df)[0]
                probabilities = trained_model.predict_proba(input_df)[0]
                
                st.write("### 📢 Kết quả đánh giá phân hệ:")
                if prediction == 1:
                    st.error(f"🚨 **CẢNH BÁO NGUY HIỂM:** Giao dịch có dấu hiệu gian lận/rủi ro cao! (Xác suất rủi ro: {probabilities[1]*100:.2f}%)")
                else:
                    st.success(f"✅ **AN TOÀN:** Giao dịch được phân loại là Bình Thường. (Xác suất an toàn: {probabilities[0]*100:.2f}%)")
                    
        else:
            st.write("#### 📥 Nhập danh sách giao dịch tổng hợp")
            st.caption("Yêu cầu định dạng tệp tải lên bắt buộc phải chứa đầy đủ 14 trường thông tin: từ `X_1` đến `X_14`.")
            
            batch_file = st.file_uploader("Tải tệp danh sách cần chấm điểm rủi ro", type=["csv", "xlsx"], key="batch_uploader")
            
            if batch_file is not None:
                try:
                    if batch_file.name.endswith('.csv'):
                        input_batch_df = pd.read_csv(batch_file)
                    else:
                        input_batch_df = pd.read_excel(batch_file)
                        
                    # Kiểm tra xem cấu trúc dữ liệu tải lên có khớp không
                    missing_features = [col for col in features if col not in input_batch_df.columns]
                    
                    if missing_features:
                        st.error(f"Tệp tải lên không hợp lệ. Thiết bị hệ thống phát hiện thiếu các cột sau: {missing_features}")
                    else:
                        # Thực hiện dự báo hàng loạt mà không huấn luyện lại
                        X_batch = input_batch_df[features]
                        batch_preds = trained_model.predict(X_batch)
                        batch_probs = trained_model.predict_proba(X_batch)[:, 1]
                        
                        # Gán nhãn kết quả trực tiếp vào DataFrame mới đầu ra
                        output_df = input_batch_df.copy()
                        output_df['Du_bao_Ket_Qua'] = batch_preds
                        output_df['Xac_Suat_Gian_Lan'] = batch_probs
                        output_df['Trang_Thai_Nhan_Dien'] = output_df['Du_bao_Ket_Qua'].map({0: 'Bình thường', 1: 'Nguy cơ gian lận'})
                        
                        st.write("#### 📋 Kết quả phân tích hàng loạt mẫu")
                        st.dataframe(output_df[['Trang_Thai_Nhan_Dien', 'Xac_Suat_Gian_Lan'] + features], use_container_width=True)
                        
                        # Xuất file kết quả dự báo ra cho người dùng tải xuống
                        csv_buffer = io.StringIO()
                        output_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Tải xuống bảng kết quả phân tích (.CSV)",
                            data=csv_data,
                            file_name="ket_qua_phat_hien_gian_lan_hang_loat.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Xảy ra lỗi trong tiến trình xử lý dữ liệu hàng loạt: {e}")
