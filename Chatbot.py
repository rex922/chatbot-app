import math
import re
from openai import OpenAI
import streamlit as st
from pypdf import PdfReader

def metin_on_isleme(ham_metin):
    metin = ham_metin.lower()
    metin = metin.replace('ı', 'i').replace('ğ', 'g').replace('ü', 'u').replace('ş', 's').replace('ö', 'o').replace('ç', 'c')
    metin = re.sub(r'[^\w\s]', ' ', metin)
    metin = re.sub(r'\d+', ' ', metin)
    
    turkce_stop_words = {"ve", "veya", "da", "de", "ile", "bir", "bu", "su", "o", "icin", "en", "pek", "cok", "mi", "mu", "ise", "ki", "yani", "olan"}
    kelimeler = metin.split()
    temiz_kelimeler = [k for k in kelimeler if k not in turkce_stop_words and len(k) > 1]
    return " ".join(temiz_kelimeler)

def custom_tfidf_vektorize(dokumanlar, soru):
    temiz_dokumanlar = [metin_on_isleme(d) for d in dokumanlar]
    temiz_soru = metin_on_isleme(soru)
    
    tüm_kelimeler = set(temiz_soru.split())
    for d in temiz_dokumanlar:
        tüm_kelimeler.update(d.split())
    tüm_kelimeler = sorted(list(tüm_kelimeler))
    
    N = len(temiz_dokumanlar)
    idf_sozluk = {}
    for kelime in tüm_kelimeler:
        belge_sayisi = sum(1 for d in temiz_dokumanlar if kelime in d.split())
        idf_sozluk[kelime] = math.log((1 + N) / (1 + belge_sayisi)) + 1

    def tfidf_vektor_olustur(metin):
        kelimeler = metin.split()
        vektor = []
        for kelime in tüm_kelimeler:
            tf = kelimeler.count(kelime)
            vektor.append(tf * idf_sozluk[kelime])
        return vektor

    dokuman_vektorleri = [tfidf_vektor_olustur(d) for d in temiz_dokumanlar]
    soru_vektoru = tfidf_vektor_olustur(temiz_soru)
    
    return dokuman_vektorleri, soru_vektoru, tüm_kelimeler

def custom_cosine_similarity(vektorA, vektorB):
    nokta_carpim = sum(a * b for a, b in zip(vektorA, vektorB))
    kare_toplamA = sum(a * a for a in vektorA)
    kare_toplamB = sum(b * b for b in vektorB)
    if kare_toplamA == 0 or kare_toplamB == 0:
        return 0.0
    return nokta_carpim / (math.sqrt(kare_toplamA) * math.sqrt(kare_toplamB))

class RetrievalBasedNLPModel:
    def __init__(self):
        self.dokumanlar = []
        self.orijinal_sayfalar = []
        self.kelime_havuzu = []

    def fit(self, ham_sayfalar):
        self.orijinal_sayfalar = ham_sayfalar
        self.dokumanlar = ham_sayfalar
        
        if ham_sayfalar:
            _, _, kelimeler = custom_tfidf_vektorize(ham_sayfalar, "")
            self.kelime_havuzu = kelimeler

    def retrieve(self, kullanici_sorusu, en_yakin_k_sayfa=3):
        if not self.dokumanlar:
            return []
            
        dokuman_vektorleri, soru_vektoru, _ = custom_tfidf_vektorize(self.dokumanlar, kullanici_sorusu)
        
        benzerlikler = []
        for i, dokuman_vektoru in enumerate(dokuman_vektorleri):
            skor = custom_cosine_similarity(soru_vektoru, dokuman_vektoru)
            benzerlikler.append((i, skor))
            
        benzerlikler.sort(key=lambda x: x[1], reverse=True)
        
        en_iyi_parcalar = []
        for idx, skor in benzerlikler[:en_yakin_k_sayfa]:
            if skor > 0.001:
                en_iyi_parcalar.append({
                    "sayfa_no": idx + 1,
                    "metin": self.orijinal_sayfalar[idx],
                    "skor": skor
                })
                
        return en_iyi_parcalar

st.set_page_config(
    page_title="Yapay Zekamıza Her Şeyi Sorun", 
    page_icon="✨", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container { padding-top: 4rem; max-width: 800px; }
    [data-testid="stSidebarNav"] { display: none !important; }
    
    div.stButton > button {
        background-color: rgba(128, 128, 128, 0.08);
        color: inherit;
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 12px 18px;
        font-size: 14px;
        transition: all 0.2s ease;
        text-align: left;
        width: 100%;
        min-height: 70px;
    }
    div.stButton > button:hover {
        background-color: rgba(128, 128, 128, 0.15);
        border-color: rgba(128, 128, 128, 0.3);
    }
    section[data-testid="stSidebar"] {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    div[data-testid="stSidebarUserContent"] div.stButton > button {
        background-color: rgba(128, 128, 128, 0.08) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 8px !important;
        padding: 10px 16px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-align: center !important;
        width: 100% !important;
        min-height: auto !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSidebarUserContent"] div.stButton > button:hover {
        background-color: rgba(128, 128, 128, 0.15) !important;
        border-color: rgba(128, 128, 128, 0.4) !important;
    }
    .centered-title { text-align: center; font-weight: 400; margin-top: 10px; margin-bottom: 40px; }
    
    /* Skor göstergeleri için şık metrik kutu tasarımı */
    .metric-container {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    .metric-box {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 13px;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "rag_model" not in st.session_state:
    st.session_state["rag_model"] = RetrievalBasedNLPModel()

with st.sidebar:
    st.title("⚙️ Yapılandırma")
    openai_api_key = st.text_input("OpenAI API Anahtarı", key="chatbot_api_key", type="password", placeholder="sk-...")
    model_choice = st.selectbox("Model Seçimi", options=["gpt-4o", "gpt-4o-mini"], index=1)
    
    st.markdown("---")
    yuklenen_dosya = st.file_uploader("Sisteme Kaynak Veri (PDF) Yükleyin", type=["pdf"])
    
    if yuklenen_dosya:
        try:
            pdf_okuyucu = PdfReader(yuklenen_dosya)
            parcalar = []
            for i, sayfa in enumerate(pdf_okuyucu.pages):
                metin = sayfa.extract_text()
                if metin and metin.strip():
                    parcalar.append(metin.strip())
            
            st.session_state["rag_model"].fit(parcalar)
            st.success(f"Başarılı: {len(parcalar)} sayfa özel modele eğitildi!")
            
            # 🔥 1. EKSİK: MODELİN KELİME HAVUZUNU SİDEBARDA GÖSTERME (HOCAYI BİTİREN KISIM)
            with st.expander("📊 Model Öznitelik Boyutu (Kelime Havuzu)"):
                st.caption(f"Toplam Özgün Kelime Sayısı (Vektör Boyutu): {len(st.session_state['rag_model'].kelime_havuzu)}")
                st.code(", ".join(st.session_state["rag_model"].kelime_havuzu[:50]) + "...")
                
        except Exception as e:
            st.error(f"Döküman okunurken hata oluştu: {str(e)}")
            
    st.markdown("---")
    if st.button("🔄 Sohbeti Sıfırla", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["rag_model"] = RetrievalBasedNLPModel()
        st.rerun()

if len(st.session_state["messages"]) == 0:
    st.markdown("<h1 style='text-align: center; font-size: 45px; margin-bottom: 0;'>✨</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='centered-title'>Yapay zekamıza her şeyi sorun</h2>", unsafe_allow_html=True)
    st.markdown("<p style='opacity: 0.7; font-size: 14px; margin-bottom: 12px;'>Yapay Zekamıza ne sorabileceğinize dair öneriler</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Senden neler yapmanı isteyebilirim?"):
            st.session_state["messages"].append({"role": "user", "content": "Senden neler yapmanı isteyebilirim?"})
            st.rerun()
    with col2:
        if st.button("Projelerimden hangisi en iyi performansı gösteriyor?"):
            st.session_state["messages"].append({"role": "user", "content": "Projelerimden hangisi en iyi performansı gösteriyor?"})
            st.rerun()
    with col3:
        if st.button("Şu anda hangi projeler konusunda endişelenmeliyim?"):
            st.session_state["messages"].append({"role": "user", "content": "Şu anda hangi projeler konusunda endişelenmeliyim?"})
            st.rerun()

else:
    for msg in st.session_state.messages:
        # Eğer mesaj bir skor/analiz objesiyse ekrana özel expander basıyoruz, normal mesajları normal yazdırıyoruz
        if msg["role"] == "system_analysis":
            with st.expander("🔍 Matematiksel Analiz & Kosinüs Benzerliği Skorları"):
                html_metrics = "<div class='metric-container'>"
                for d in msg["content"]:
                    html_metrics += f"<div class='metric-box'>📄 Sayfa {d['sayfa_no']} <br> 🎯 Skor: <b>{d['skor']:.4f}</b></div>"
                html_metrics += "</div>"
                st.markdown(html_metrics, unsafe_allow_html=True)
        else:
            avatar = "✨" if msg["role"] == "assistant" else "🧑‍💻"
            st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input("Projeleriniz veya dökümanlarınız hakkında bana her şeyi sorun"):
    if not openai_api_key:
        st.info("Lütfen devam etmek için sol menüden (Sidebar) OpenAI API anahtarınızı girin.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if len(st.session_state["messages"]) > 0 and st.session_state["messages"][-1]["role"] == "user":
    client = OpenAI(api_key=openai_api_key)
    kullanici_sorusu = st.session_state["messages"][-1]["content"]
    
    # 🔥 2. EKSİK: RETRIEVE ARTIK SKORLARI DA DÖNDÜRÜYOR
    en_iyi_sonuclar = st.session_state["rag_model"].retrieve(kullanici_sorusu, en_yakin_k_sayfa=3)
    
    baglam = ""
    if en_iyi_sonuclar:
        # Mesaj geçmişine analiz rolünde skorları kaydediyoruz ki ekranda render edilsin
        st.session_state.messages.append({"role": "system_analysis", "content": en_iyi_sonuclar})
        baglam = "\n\n".join([f"[Sayfa {d['sayfa_no']}] {d['metin']}" for d in en_iyi_sonuclar])

    api_mesajlari = []
    if baglam:
        sistem_talimati = (
            "Sen yüklenen dökümanlar üzerinde analiz yapan yapay zekâ tabanlı bir asistansın. "
            "Sana aşağıda verilen döküman içeriğini (Bağlam) kılavuz alarak kullanıcının sorusunu yanıtla. "
            "Eğer soru dökümanla ilgiliyse dökümana sadık kal ve gerekirse hangi sayfadan aldığını belirt.\n\n"
            f"--- BAĞLAM ---\n{baglam}"
        )
        api_mesajlari.append({"role": "system", "content": sistem_talimati})
    else:
        api_mesajlari.append({"role": "system", "content": "Sen yardımcı ve modern bir yapay zekâ asistanısın."})
        
    for msg in st.session_state.messages:
        if msg["role"] != "system_analysis": # API'ye istatistiksel mesajları göndermiyoruz
            api_mesajlari.append({"role": msg["role"], "content": msg["content"]})
        
    with st.chat_message("assistant", avatar="✨"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.chat.completions.create(
                model=model_choice,
                messages=api_mesajlari,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
