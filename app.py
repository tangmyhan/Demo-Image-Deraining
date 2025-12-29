import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
import numpy as np
from streamlit_image_comparison import image_comparison

# ==========================================
# CẤU HÌNH
# ==========================================
# Hugging Face Free CPU hơi yếu, ta giảm Max Size xuống một chút để không bị đơ
MAX_SIZE = 480 
MODEL_FILENAME = "best_model.pth"

# Cấu hình Model (KHỚP VỚI FILE TRAIN)
NUM_BLOCKS = [3, 4, 4, 6]
NUM_HEADS = [1, 2, 2, 4]
CHANNELS = [32, 64, 128, 256]
EXPANSION_FACTOR = 2.66
NUM_REFINEMENT = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ĐỊNH NGHĨA MODEL RESTORMER ---
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(b, self.num_heads, -1, h * w), [q, k, v])
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)
    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1).contiguous().reshape(b, c, h, w))
        return x

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False), nn.PixelUnshuffle(2))
    def forward(self, x): return self.body(x)

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False), nn.PixelShuffle(2))
    def forward(self, x): return self.body(x)

class Restormer(nn.Module):
    def __init__(self, num_blocks, num_heads, channels, num_refinement, expansion_factor):
        super(Restormer, self).__init__()
        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(c, h, expansion_factor) for _ in range(b)]) for b, h, c in zip(num_blocks, num_heads, channels)])
        self.downs = nn.ModuleList([DownSample(c) for c in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(c) for c in list(reversed(channels))[:-1]])
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i-1], kernel_size=1, bias=False) for i in reversed(range(2, len(channels)))])
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor) for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor) for _ in range(num_blocks[1])]))
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])]))
        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))
        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        return self.output(fr) + x

# ==========================================
# CÁC HÀM XỬ LÝ
# ==========================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILENAME):
        return None
    
    # Khởi tạo model
    model = Restormer(NUM_BLOCKS, NUM_HEADS, CHANNELS, NUM_REFINEMENT, EXPANSION_FACTOR)
    
    # FIX LỖI Weights Only tại đây
    checkpoint = torch.load(MODEL_FILENAME, map_location=DEVICE, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()
    return model

def process_image(img_pil, model):
    w_orig, h_orig = img_pil.size
    
    # Resize nếu quá to
    if w_orig > MAX_SIZE or h_orig > MAX_SIZE:
        scale = min(MAX_SIZE/w_orig, MAX_SIZE/h_orig)
        new_w = int(w_orig * scale)
        new_h = int(h_orig * scale)
        img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
    
    w, h = img_pil.size
    
    # Pad ảnh
    new_w = ((w + 7) // 8) * 8
    new_h = ((h + 7) // 8) * 8
    pad_w = new_w - w
    pad_h = new_h - h
    
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(DEVICE)
    img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), 'reflect')
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        
    # Post-process
    output = output[:, :, :h, :w]
    output = torch.clamp(output, 0, 1)
    output_pil = TF.to_pil_image(output.squeeze(0).cpu())
    
    return img_pil, output_pil

# ==========================================
# GIAO DIỆN
# ==========================================

st.set_page_config(page_title="Restormer Deraining", layout="centered")
st.title("🌧️ Khử Mưa - Restormer AI")
st.caption(f"Running on: {DEVICE} (Max size: {MAX_SIZE}px)")

# Load Model
try:
    model = load_model()
except Exception as e:
    st.error(f"Lỗi load model: {e}")
    model = None

if model is None:
    st.warning(f"⚠️ Đang tìm file `{MODEL_FILENAME}`... Vui lòng upload file model vào Space.")
else:
    uploaded_file = st.file_uploader("Upload ảnh mưa:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert('RGB')
        
        if st.button("🚀 Bắt đầu khử mưa"):
            with st.spinner("Model đang chạy..."):
                try:
                    img_in, img_out = process_image(input_image, model)
                    
                    st.success("Xử lý xong!")
                    
                    # Component so sánh
                    image_comparison(
                        img1=img_in,
                        img2=img_out,
                        label1="Ảnh Gốc",
                        label2="Ảnh Sạch",
                        width=700,
                        starting_position=50,
                        show_labels=True,
                        make_responsive=True,
                        in_memory=True
                    )
                    
                    # Nút tải về
                    from io import BytesIO
                    buf = BytesIO()
                    img_out.save(buf, format="PNG")
                    st.download_button("📥 Tải ảnh về", buf.getvalue(), "clean_image.png", "image/png")
                    
                except Exception as e:
                    st.error(f"Lỗi xử lý: {e}")