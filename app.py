import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from streamlit_image_comparison import image_comparison
from io import BytesIO

try:
    from restormer_model import Restormer
    from pix2pix_model import UnetGenerator # Đổi tên file model.py của pix2pix thành pix2pix_model.py
except ImportError:
    st.error("⚠️ Lỗi: Không tìm thấy file 'restormer_model.py' hoặc 'pix2pix_model.py'. Vui lòng kiểm tra lại cấu trúc thư mục.")
    st.stop()

# ==========================================
# CẤU HÌNH CHUNG
# ==========================================
st.set_page_config(page_title="Image Deraining Zoo", layout="wide", page_icon="🌧️")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("🌧️ Image Deraining Model Zoo")
st.markdown("Demo so sánh các mô hình xóa mưa: **Restormer** (Transformer-based) và **Pix2Pix** (GAN-based).")

# Sidebar chọn Model
st.sidebar.header("⚙️ Cấu hình")
model_choice = st.sidebar.selectbox("Chọn Model:", ["Restormer", "Pix2Pix"])

# ==========================================
# 1. LOGIC CHO RESTORMER
# ==========================================
def setup_restormer():
    # Config Restormer
    MAX_SIZE = 480 
    MODEL_FILENAME = "best_model.pth" # Đường dẫn file model Restormer
    
    # Model Params
    NUM_BLOCKS = [3, 4, 4, 6]
    NUM_HEADS = [1, 2, 2, 4]
    CHANNELS = [32, 64, 128, 256]
    EXPANSION_FACTOR = 2.66
    NUM_REFINEMENT = 2

    @st.cache_resource
    def load_restormer_model():
        path = MODEL_FILENAME
        if not os.path.exists(path):
            st.error(f"Không tìm thấy file model Restormer tại: {path}")
            return None
        
        model = Restormer(NUM_BLOCKS, NUM_HEADS, CHANNELS, NUM_REFINEMENT, EXPANSION_FACTOR)
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(DEVICE)
        model.eval()
        return model

    def process_restormer(img_pil, model):
        w_orig, h_orig = img_pil.size
        
        # Resize để tránh OOM trên CPU
        if w_orig > MAX_SIZE or h_orig > MAX_SIZE:
            scale = min(MAX_SIZE/w_orig, MAX_SIZE/h_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
        
        w, h = img_pil.size
        
        # Padding chia hết cho 8
        new_w = ((w + 7) // 8) * 8
        new_h = ((h + 7) // 8) * 8
        pad_w = new_w - w
        pad_h = new_h - h
        
        img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(DEVICE)
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')
        
        with torch.no_grad():
            output = model(img_tensor)
            
        output = output[:, :, :h, :w]
        output = torch.clamp(output, 0, 1)
        output_pil = TF.to_pil_image(output.squeeze(0).cpu())
        
        return img_pil, output_pil # Trả về cả ảnh gốc (đã resize) và ảnh kết quả

    return load_restormer_model, process_restormer

# ==========================================
# 2. LOGIC CHO PIX2PIX
# ==========================================
def setup_pix2pix():
    MODEL_PATH = "generator_best.pth"

    @st.cache_resource
    def load_pix2pix_model():
        if not os.path.exists(MODEL_PATH):
            st.error(f"Không tìm thấy file model Pix2Pix tại: {MODEL_PATH}")
            return None
            
        model = UnetGenerator().to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [-1,1]
    ])

    def pad_to_multiple(img, multiple=256):
        _, _, h, w = img.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        # Pix2Pix model thường yêu cầu kích thước cụ thể hoặc chia hết cho số lớn hơn (như 256)
        if pad_h == multiple: pad_h = 0
        if pad_w == multiple: pad_w = 0
        img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        return img_padded, h, w


    def process_pix2pix(image_pil, model):
        # Resize nhẹ nếu ảnh quá lớn
        max_size = 512
        if max(image_pil.size) > max_size:
            ratio = max_size / max(image_pil.size)
            new_size = (int(image_pil.size[0] * ratio), int(image_pil.size[1] * ratio))
            image_pil = image_pil.resize(new_size, Image.BILINEAR)

        img = np.array(image_pil).astype(np.float32) / 255.0
        tensor = to_tensor(img).unsqueeze(0).to(DEVICE)

        # Pad chia hết cho 256 (đã sửa ở bước trước)
        tensor_pad, h, w = pad_to_multiple(tensor, multiple=256) 

        with torch.no_grad():
            fake_pad = model(tensor_pad)

        fake = fake_pad[:, :, :h, :w]
        fake = torch.clamp((fake + 1) / 2, 0, 1)

        fake_img = fake.squeeze(0).cpu().permute(1, 2, 0).numpy()
        fake_img = (fake_img * 255).astype(np.uint8)
        
        return image_pil, Image.fromarray(fake_img)

    # --- QUAN TRỌNG: BẠN ĐANG THIẾU DÒNG NÀY HOẶC NÓ BỊ THỤT VÀO TRONG ---
    return load_pix2pix_model, process_pix2pix

# ==========================================
# 3. LUỒNG XỬ LÝ CHÍNH
# ==========================================

# Upload ảnh
uploaded_file = st.file_uploader("📤 Upload ảnh có mưa (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    # Hiển thị ảnh gốc
    st.sidebar.image(input_image, caption="Ảnh gốc", use_container_width=True)
    
    # Load model dựa trên lựa chọn
    if model_choice == "Restormer":
        load_fn, process_fn = setup_restormer()
        st.info("Đang sử dụng model: **Restormer**")
    else:
        load_fn, process_fn = setup_pix2pix()
        st.info("Đang sử dụng model: **Pix2Pix**")

    # Load Model (Cached)
    model = load_fn()

    if model is not None:
        # Nút chạy inference
        if st.button(f"🚀 Chạy khử mưa với {model_choice}"):
            with st.spinner(f"Đang xử lý bằng {model_choice}... vui lòng chờ."):
                try:
                    # Xử lý ảnh
                    # Cả 2 hàm process đều trả về (ảnh input đã resize, ảnh output PIL)
                    img_in_final, img_out_final = process_fn(input_image, model)
                    
                    st.success("Xử lý hoàn tất!")
                    st.subheader("🔍 So sánh kết quả")

                    # Image Comparison Slider
                    image_comparison(
                        img1=img_in_final,
                        img2=img_out_final,
                        label1="Input (Mưa)",
                        label2=f"Output ({model_choice})",
                        width=800,
                        starting_position=50,
                        show_labels=True,
                        make_responsive=True,
                        in_memory=True
                    )

                    # Nút Download
                    buf = BytesIO()
                    img_out_final.save(buf, format="PNG")
                    st.download_button(
                        label=f"⬇️ Tải ảnh kết quả ({model_choice})",
                        data=buf.getvalue(),
                        file_name=f"derained_{model_choice}.png",
                        mime="image/png"
                    )

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
                    st.error("Gợi ý: Nếu lỗi CUDA OOM, hãy thử ảnh nhỏ hơn hoặc chuyển sang CPU.")

else:
    st.info("👆 Vui lòng upload một ảnh để bắt đầu.")

# Footer
st.markdown("---")
st.markdown(f"Running on **{DEVICE}** | Code combined for multi-model demo.")