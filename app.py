import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from io import BytesIO
from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="Deraining", layout="wide", page_icon="🌧️")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from restormer_model import Restormer
    from pix2pix_model import UnetGenerator
except ImportError:
    st.error("⚠️ Lỗi: Thiếu file 'restormer_model.py' hoặc 'pix2pix_model.py'.")
    st.stop()

# LOAD & XỬ LÝ RESTORMER
@st.cache_resource
def load_restormer():
    path = "best_model.pth"
    if not os.path.exists(path): return None
    
    # Config
    model = Restormer(num_blocks=[3, 4, 4, 6], num_heads=[1, 2, 2, 4], 
                      channels=[32, 64, 128, 256], num_refinement=2, expansion_factor=2.66)
    
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

def run_restormer(img_pil, model):
    w, h = img_pil.size
    
    # Resize nếu quá to 
    max_size = 512
    if max(w, h) > max_size:
        scale = min(max_size/w, max_size/h)
        img_pil = img_pil.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
        w, h = img_pil.size

    # Pad chia hết cho 8
    new_w = ((w + 7) // 8) * 8
    new_h = ((h + 7) // 8) * 8
    pad_w, pad_h = new_w - w, new_h - h
    
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(DEVICE)
    img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')
    
    with torch.no_grad():
        output = model(img_tensor)
        
    output = output[:, :, :h, :w]
    output = torch.clamp(output, 0, 1)
    return TF.to_pil_image(output.squeeze(0).cpu()), img_pil # Trả về cả input đã resize

#  LOAD & XỬ LÝ PIX2PIX
@st.cache_resource
def load_pix2pix():
    path = "generator_best.pth" 
    if not os.path.exists(path): return None
    
    model = UnetGenerator().to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def run_pix2pix(img_pil, model):
    w, h = img_pil.size
    max_size = 512
    if max(w, h) > max_size:
        scale = min(max_size/w, max_size/h)
        img_pil = img_pil.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    img = np.array(img_pil).astype(np.float32) / 255.0
    tensor = to_tensor(img).unsqueeze(0).to(DEVICE)
    
    # Pad chia hết cho 256
    multiple = 256
    pad_h = (multiple - tensor.shape[2] % multiple) % multiple
    pad_w = (multiple - tensor.shape[3] % multiple) % multiple
    
    tensor_pad = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        fake_pad = model(tensor_pad)

    fake = fake_pad[:, :, :tensor.shape[2], :tensor.shape[3]]
    fake = torch.clamp((fake + 1) / 2, 0, 1)
    
    fake_img = fake.squeeze(0).cpu().permute(1, 2, 0).numpy()
    fake_img = (fake_img * 255).astype(np.uint8)
    
    return Image.fromarray(fake_img), img_pil # Trả về output và input (đã resize)

# GIAO DIỆN CHÍNH
# 
st.title("🌧️ Removing rain from single images")
# st.markdown("### So sánh trực tiếp hiệu quả khử mưa giữa hai kiến trúc mạng.")
st.markdown(
    """
    **Mô hình:** Restormer vs. Pix2Pix 
    **Chức năng:** Xóa mưa từ một ảnh đầu vào  
    **So sánh:** Kéo thanh trượt để so sánh Input và Output
    """
)
# --- Load Models ---
with st.spinner("Đang tải model..."):
    model_res = load_restormer()
    model_pix = load_pix2pix()

if model_res is None or model_pix is None:
    st.error("⚠️ Không tìm thấy file model `.pth`. Vui lòng kiểm tra lại thư mục.")
else:
    st.success(f"Đã tải xong 2 model trên thiết bị: **{DEVICE}**")

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload ảnh mưa (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
        
        # Nút chạy
        if st.button("Bắt đầu khử mưa", type="primary"):
            
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # 1. Chạy Restormer
                status_text.text("🤖 Đang chạy Restormer...")
                out_res, input_res_resized = run_restormer(raw_image, model_res)
                progress_bar.progress(50)
                
                # 2. Chạy Pix2Pix
                status_text.text("🎨 Đang chạy Pix2Pix...")
                out_pix, input_pix_resized = run_pix2pix(raw_image, model_pix)
                progress_bar.progress(100)
                
                status_text.empty()
                progress_bar.empty()
                
                # --- HIỂN THỊ KẾT QUẢ ---
                st.markdown("---")
                st.subheader("Kết Quả So Sánh")
   
                # Restormer
                st.markdown("### **Restormer** (Transformer)")
                image_comparison(
                    img1=input_res_resized,
                    img2=out_res,
                    label1="Input",
                    label2="Restormer",
                    width=700, # Độ rộng tối đa trong cột
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True
                )
                # Download Restormer
                buf1 = BytesIO()
                out_res.save(buf1, format="PNG")
                st.download_button("📥 Tải ảnh về", buf1.getvalue(), "restormer.png", "image/png")
                
                # Pix2Pix
                st.markdown("### **Pix2Pix** (GAN)")
                image_comparison(
                    img1=input_pix_resized,
                    img2=out_pix,
                    label1="Input",
                    label2="Pix2Pix",
                    width=700,
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True
                )
                # Download Pix2Pix
                buf2 = BytesIO()
                out_pix.save(buf2, format="PNG")
                st.download_button("📥 Tải ảnh về", buf2.getvalue(), "pix2pix.png", "image/png")

            except Exception as e:
                st.error(f"Lỗi xử lý: {e}")
