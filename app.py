import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# İstemci arayüzü ayarları
st.set_page_config(page_title="Kalp Hastalığı Teşhis Sistemi", layout="wide")

# Modeli eğitme fonksiyonu
@st.cache_resource
def train_model():
    df = pd.read_csv('Heart_disease_cleveland_new.csv')
    
    # Veri seti sütun adını 'target' olarak eşitleme
    if 'target' not in df.columns:
        df.rename(columns={df.columns[-1]: 'target'}, inplace=True)
        
    # SFS ile seçilen en iyi 5 özellik
    selected_features = ['cp', 'restecg', 'slope', 'ca', 'thal']
    X = df[selected_features]
    y = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Model eğitimi
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, X

model, X_train = train_model()

# Başlık ve Giriş
st.title("Açıklanabilir Yapay Zeka (XAI) Destekli Kalp Hastalığı Teşhis Sistemi")
st.markdown("""
Bu sistem, hekimlere karar destek aracı olarak tasarlanmıştır. "Kara Kutu" (Black-Box) problemini çözmek ve tanı sürecini hızlandırmak amacıyla iki temel akademik metodoloji kullanılmıştır:
""")

# SFS Açıklaması
st.info("""
**Neden Sadece Bu 5 Özellik Kullanılıyor?**  
Makine öğrenmesi modellerine tüm hasta verilerini vermek, karmaşıklığı (gürültüyü) artırarak modelin yavaşlamasına ve hata yapmasına neden olabilir. Bu projede **SFS (Sequential Forward Selection)** algoritması kullanılarak 13 klinik özellik taranmış ve modelin hata payını en aza indiren en kritik 5 özellik otomatik olarak seçilmiştir.
""")

# Yan Menü - Hasta Veri Girişi
st.sidebar.header("Hasta Veri Girişi")

# Seçeneklerin tıbbi açıklamaları
cp_options = {0: "Tipik Anjina", 1: "Atipik Anjina", 2: "Anjina Dışı Ağrı", 3: "Asemptomatik (Belirti yok)"}
restecg_options = {0: "Normal", 1: "ST-T Dalga Anormalliği", 2: "Sol Ventriküler Hipertrofi"}
slope_options = {0: "Yukarı Eğimli", 1: "Düz", 2: "Aşağı Eğimli"}
thal_options = {0: "Bilinmiyor (Null)", 1: "Normal Kan Akışı", 2: "Sabit Hata (Kalbin bir kısmında kan akışı yok)", 3: "Tersine Çevrilebilir Hata (Anormal Kan Akışı)"}

cp = st.sidebar.selectbox("Göğüs Ağrısı Tipi (cp)", options=[0, 1, 2, 3], format_func=lambda x: f"{x}: {cp_options[x]}")
restecg = st.sidebar.selectbox("Dinlenik EKG Sonuçları (restecg)", options=[0, 1, 2], format_func=lambda x: f"{x}: {restecg_options[x]}")
slope = st.sidebar.selectbox("Egzersiz ST Segment Eğimi (slope)", options=[0, 1, 2], format_func=lambda x: f"{x}: {slope_options[x]}")
ca = st.sidebar.slider("Tıkanmış Ana Damar Sayısı (ca)", 0, 3, 0, help="Floroskopi ile renklendirilmiş ana damar sayısı (0-3 arası)")
thal = st.sidebar.selectbox("Talasemi / Kan Akışı (thal)", options=[0, 1, 2, 3], format_func=lambda x: f"{x}: {thal_options[x]}")

# Kullanıcı girişini DataFrame'e dönüştürme
patient_data = pd.DataFrame({
    'cp': [cp],
    'restecg': [restecg],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Analiz Butonu
if st.sidebar.button("Hasta Durumunu Analiz Et", type="primary"):
    
    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0][1] * 100
    
    st.divider()
    
    col1, col2 = st.columns([1, 1.5])
    
    # Sonuç Paneli
    with col1:
        st.subheader("Klinik Sonuç:")
        if prediction == 1:
            st.error("Uyarı: Hasta kalp hastalığı riski taşıyor.")
            st.metric(label="Hesaplanan Risk Oranı", value=f"%{probability:.1f}")
        else:
            st.success("Hastanın durumu büyük ihtimalle sağlıklı, ciddi bir risk belirtisi yok.")
            st.metric(label="Hesaplanan Risk Oranı", value=f"%{probability:.1f}")

    # SHAP Paneli
    with col2:
        st.subheader("Yapay Zeka Karar Açıklaması (SHAP):")
        st.write("Bu hastanın mevcut durumu modelin kararını nasıl etkiledi?")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_data)
        
        if isinstance(shap_values, list):
            patient_shap = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            patient_shap = shap_values[0, :, 1]
        else:
            patient_shap = shap_values[0]
            
        fig, ax = plt.subplots(figsize=(6, 4))
        y_pos = np.arange(len(patient_data.columns))
        colors = ['#ff0051' if val > 0 else '#008bfb' for val in patient_shap]
        
        ax.barh(y_pos, patient_shap, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(patient_data.columns)
        ax.invert_yaxis()
        ax.set_xlabel('Özellik Etki Değeri (Kırmızı = Riski Artırır / Mavi = Riski Azaltır)')
        ax.set_title('Mevcut Hasta İçin Model Karar Açıklaması')
        
        st.pyplot(fig)
        
        # SHAP Grafiği Açıklaması
        st.caption("""
        **Grafik Nasıl Okunur?**  
        Çubukların **sağa doğru (Kırmızı)** uzaması, o klinik değerin hastanın kalp hastalığı riskini artırdığını gösterir. **Sola doğru (Mavi)** uzaması ise o değerin riski düşürdüğünü ifade eder. Çubuğun uzunluğu, o özelliğin karardaki ağırlığını (önemini) temsil eder.
        """)

# Referanslar
st.divider()
st.subheader("Referans Makaleler")
st.markdown("""
1. [Explainable Machine Learning Framework for Heart Disease Prediction Using SHAP-Based Feature Optimization](https://ieeexplore.ieee.org/document/11378622)
2. [Performance Improvement of Heart Disease Prediction by Identifying Optimal Feature Sets Using Feature Selection Technique](https://ieeexplore.ieee.org/document/9491739)
""")
