import streamlit as st
import numpy as np
import sys
import traceback
import cv2  # Cần import cv2 để dùng cv2.imdecode

# Import từ các file helper
# (Giả định ui_helpers.py nằm cùng cấp app.py)
from ui_helpers import convert_image_for_streamlit

try:
    # Import pipeline từ package lpr
    from lpr.pipeline import detect_plate_manual_steps
except ImportError:
    st.error("Lỗi: Không tìm thấy package 'lpr'.")
    st.info("Vui lòng đảm bảo 'app.py' nằm cùng cấp với thư mục 'lpr'.")
    sys.exit(1)
except Exception as e:
    st.error(f"Lỗi khi import 'lpr.pipeline': {e}")
    st.info(f"Hãy đảm bảo package 'lpr' không có lỗi.\nChi tiết: {e}")
    sys.exit(1)

# -----------------------------------------------------------------
# ======= PHẦN 1: GIAO DIỆN SIDEBAR (ĐỊNH NGHĨA WIDGETS) =======
# -----------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Trình xem các bước xử lý ảnh (bằng Streamlit)")

with st.sidebar:
    st.header("Điều khiển")

    # 1. Nút chọn ảnh
    uploaded_file = st.file_uploader(
        "Chọn một ảnh",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Ảnh sẽ được xử lý lại mỗi khi bạn thay đổi tham số."
    )

    # 2. Nút làm mới
    if st.button("✨ Xóa ảnh và Reset"):
        # Xóa các key trong bộ nhớ
        if 'image_bytes' in st.session_state:
            del st.session_state['image_bytes']
        if 'file_name' in st.session_state:
            del st.session_state['file_name']
        st.success("Đã xóa ảnh, vui lòng upload ảnh mới.")
        st.rerun()

    st.divider()

    # -------------------------------------------------
    # === THAM SỐ PIPELINE ===
    # *** QUAN TRỌNG ***
    # Các widget này được ĐỊNH NGHĨA ở đây.
    # Giá trị của chúng sẽ được ĐỌC ở "PHẦN 3"
    # -------------------------------------------------
    st.header("Tinh chỉnh Pipeline")

    st.subheader("0. Tiền xử lý")
    # Đặt key cho widget để truy cập dễ dàng
    p_resize_w = st.number_input("Resize Width", 100, 2000, 1000, key="p_resize_w")

    st.subheader("2. Gaussian Blur")
    p_gaussian_k = st.slider("Kernel Size (k x k)", 3, 11, 5, 2, key="p_gaussian_k")
    p_gaussian_sigma = st.slider("Sigma", 0.5, 5.0, 2.0, 0.1, key="p_gaussian_sigma")

    st.subheader("3. Blackhat")
    p_blackhat_kx = st.slider("Kernel Width", 3, 25, 15, 2, key="p_blackhat_kx")
    p_blackhat_ky = st.slider("Kernel Height", 3, 25, 5, 2, key="p_blackhat_ky")

    st.subheader("7. Opening")
    p_open_k = st.slider("Kernel Size (k x k)", 3, 9, 5, 2, key="p_open_k")

    st.subheader("8. Closing")
    p_close_kx = st.slider("Kernel Width", 3, 25, 11, 2, key="p_close_kx")
    p_close_ky = st.slider("Kernel Height", 3, 25, 5, 2, key="p_close_ky")

    st.subheader("9. Lọc Blob (Ngưỡng)")
    p_min_area = st.number_input("Min Area", 100, 10000, 5000, key="p_min_area")
    p_max_area = st.number_input("Max Area", 10000, 100000, 50000, key="p_max_area")
    p_min_extent = st.slider("Min Extent", 0.1, 1.0, 0.2, 0.05, key="p_min_extent")

    st.caption("Lọc tỷ lệ AR (biển dài)")
    ar1_cols = st.columns(2)
    p_ar1_min = ar1_cols[0].number_input("Min AR (dài)", 1.0, 5.0, 3.0, 0.1, key="p_ar1_min")
    p_ar1_max = ar1_cols[1].number_input("Max AR (dài)", 3.0, 10.0, 5.5, 0.1, key="p_ar1_max")

    st.caption("Lọc tỷ lệ AR (biển vuông/xe máy)")
    ar2_cols = st.columns(2)
    p_ar2_min = ar2_cols[0].number_input("Min AR (vuông)", 0.5, 2.0, 1.0, 0.1, key="p_ar2_min")
    p_ar2_max = ar2_cols[1].number_input("Max AR (vuông)", 1.0, 5.0, 2.5, 0.1, key="p_ar2_max")

# -----------------------------------------------------------------
# ======= PHẦN 2: LOGIC NẠP ẢNH (CHỈ CHẠY KHI UPLOAD MỚI) =======
# -----------------------------------------------------------------

if uploaded_file is not None:
    # Khi có file mới, đọc và lưu bytes vào bộ nhớ
    # (Việc này sẽ ghi đè ảnh cũ)
    file_bytes = uploaded_file.getvalue()
    st.session_state['image_bytes'] = file_bytes
    st.session_state['file_name'] = uploaded_file.name

# -----------------------------------------------------------------
# ======= PHẦN 3: LOGIC XỬ LÝ & HIỂN THỊ (CHẠY MỖI KHI RERUN) =======
# -----------------------------------------------------------------

st.header("Kết quả xử lý")

# Kiểm tra xem đã có ảnh trong bộ nhớ chưa
if 'image_bytes' not in st.session_state:
    st.info("Chưa có ảnh nào được xử lý. Vui lòng chọn ảnh ở thanh bên trái.")
else:
    # Nếu có ảnh, tiến hành xử lý
    file_name = st.session_state['file_name']
    image_bytes = st.session_state['image_bytes']

    try:
        # 1. Decode ảnh từ bytes (thay vì đọc từ file upload)
        file_bytes_np = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Không thể decode ảnh. File có thể bị hỏng.")
        else:
            # 2. ĐỌC CÁC GIÁ TRỊ THAM SỐ TỪ SIDEBAR
            # (Streamlit sẽ tự động lấy giá trị hiện tại của widget)
            params = {
                'resize_w': st.session_state.p_resize_w,
                'gaussian_k': st.session_state.p_gaussian_k,
                'gaussian_sigma': st.session_state.p_gaussian_sigma,
                'blackhat_kx': st.session_state.p_blackhat_kx,
                'blackhat_ky': st.session_state.p_blackhat_ky,
                'open_k': st.session_state.p_open_k,
                'close_kx': st.session_state.p_close_kx,
                'close_ky': st.session_state.p_close_ky,
                'min_area': st.session_state.p_min_area,
                'max_area': st.session_state.p_max_area,
                'ar_range_1': (st.session_state.p_ar1_min, st.session_state.p_ar1_max),
                'ar_range_2': (st.session_state.p_ar2_min, st.session_state.p_ar2_max),
                'min_extent': st.session_state.p_min_extent
            }

            # 3. Chạy pipeline với các tham số này
            with st.spinner("Đang chạy pipeline..."):
                steps_dict = detect_plate_manual_steps(img_bgr, **params)

            # 4. Hiển thị kết quả (chỉ 1 kết quả)
            st.subheader(f"🖼️ Kết quả cho: {file_name}")

            MAX_COLS = 4
            step_items = list(steps_dict.items())

            for i in range(0, len(step_items), MAX_COLS):
                cols = st.columns(MAX_COLS)
                batch = step_items[i: i + MAX_COLS]

                for j, (step_name, cv_img) in enumerate(batch):
                    with cols[j]:
                        st.caption(f"<b>{step_name}</b>", unsafe_allow_html=True)
                        img_to_show = convert_image_for_streamlit(cv_img)
                        if img_to_show is not None:
                            st.image(img_to_show, use_container_width=True)
                        else:
                            st.warning("Không có ảnh")
            st.divider()

    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi chạy pipeline: {e}")
        st.exception(traceback.format_exc())