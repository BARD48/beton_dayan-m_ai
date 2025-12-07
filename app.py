import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import plotly.graph_objects as go

# --- 1. MODEL SINIFINI TEKRAR TANIMLA ---
class ANFIS(nn.Module):
    def __init__(self, n_inputs, n_rules):
        super(ANFIS, self).__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.c = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.sigma = nn.Parameter(torch.abs(torch.randn(n_rules, n_inputs)) + 0.1)
        self.consequent_weights = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.consequent_bias = nn.Parameter(torch.randn(n_rules, 1))

    def forward(self, x):
        x_expanded = x.unsqueeze(1) 
        membership = torch.exp(-0.5 * ((x_expanded - self.c) / self.sigma) ** 2)
        w = torch.prod(membership, dim=2, keepdim=True)
        w_sum = torch.sum(w, dim=1, keepdim=True)
        w_norm = w / (w_sum + 1e-8)
        rule_output = (x_expanded * self.consequent_weights.unsqueeze(0)).sum(dim=2, keepdim=True) + self.consequent_bias.unsqueeze(0)
        weighted_output = w_norm * rule_output
        final_output = torch.sum(weighted_output, dim=1)
        return final_output

# --- 2. AYARLAR VE YÜKLEME ---
st.set_page_config(page_title="Beton Dayanım Tahmini", page_icon="🏗️")

@st.cache_resource
def load_model_and_scalers():
    # Model yapısını kur (Eğitimdeki parametrelerle aynı olmalı: 2 girdi, 4 kural)
    model = ANFIS(n_inputs=2, n_rules=4)
    # Kaydedilmiş ağırlıkları yükle
    model.load_state_dict(torch.load('anfis_model_agirliklari.pth'))
    model.eval() # Test moduna al
    
    # Scaler'ları yükle
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    return model, scaler_x, scaler_y

try:
    model, scaler_x, scaler_y = load_model_and_scalers()
except FileNotFoundError:
    st.error("Model dosyaları bulunamadı! Lütfen önce eğitim kodunu çalıştırıp .pth ve .pkl dosyalarını oluşturun.")
    st.stop()

# --- 3. ARAYÜZ TASARIMI ---
st.title("🏗️ Beton Basınç Dayanımı Tahmini (AI-SonReb)")
st.markdown("Bu uygulama, **Ultrasonik Ses Hızı (UPV)** ve **Geri Sıçrama Sayısı (RN)** kullanarak betonun dayanımını tahmin eder.")

# Yan panel veya üst kısım girişleri
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Ölçüm Verileri")
    
    # UPV Girişi (Zaten ondalıklıydı)
    upv_input = st.number_input(
        "Ultrasonik Ses Hızı (km/s)", 
        min_value=3.0, 
        max_value=6.0, 
        value=4.50, 
        step=0.01,
        format="%.2f"
    )
    
    # RN Girişi (GÜNCELLENDİ: Artık ondalıklı sayı kabul ediyor)
    rn_input = st.number_input(
        "Geri Sıçrama Sayısı (RN)", 
        min_value=10.0, 
        max_value=70.0, 
        value=30.0,  # Varsayılan değer float yapıldı (30.0)
        step=0.1,    # Adım aralığı 0.1 yapıldı (28.5 girebilmek için)
        format="%.1f" # Ekranda tek basamaklı ondalık göster (Örn: 28.5)
    )

# Tahmin Butonu
if st.button("HESAPLA", type="primary"):
    # 1. Veriyi Hazırla (Normalize Et)
    input_data = np.array([[upv_input, rn_input]])
    input_scaled = scaler_x.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # 2. Tahmin Yap
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
    
    # 3. Sonucu Gerçek Değere Çevir (De-normalize)
    prediction_real = scaler_y.inverse_transform(prediction_scaled.numpy())
    sonuc = prediction_real[0][0]

    # --- 4. SONUÇ GÖSTERİMİ ---
    with col2:
        st.subheader("💡 Sonuç")
        
        # Gösterge Grafiği (Gauge Chart)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sonuc,
            title = {'text': "Basınç Dayanımı (MPa)"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 80]}, 
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "#ff9999"}, # Zayıf
                    {'range': [20, 40], 'color': "#ffff99"}, # Orta
                    {'range': [40, 80], 'color': "#99ff99"}], # Güçlü
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': sonuc}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    # Yorum Satırı
    if sonuc < 20:
        st.error(f"Tahmin: {sonuc:.2f} MPa - Beton kalitesi DÜŞÜK görünüyor.")
    elif sonuc < 40:
        st.warning(f"Tahmin: {sonuc:.2f} MPa - Beton kalitesi ORTA seviyede.")
    else:
        st.success(f"Tahmin: {sonuc:.2f} MPa - Beton kalitesi YÜKSEK.")

st.markdown("---")
st.caption("Bu model ANFIS mimarisi kullanılarak PyTorch ile geliştirilmiştir.")