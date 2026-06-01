import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "dokuman_metinleri" not in st.session_state:
    st.session_state["dokuman_metinleri"] = []

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
                    parcalar.append(f"[Sayfa {i+1}] {metin.strip()}")
            st.session_state["dokuman_metinleri"] = parcalar
            st.success(f"Başarılı: {len(parcalar)} sayfa sisteme indekslendi!")
        except Exception as e:
            st.error(f"Döküman okunurken hata oluştu: {str(e)}")
            
    st.markdown("---")
    if st.button("🔄 Sohbeti Sıfırla", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["dokuman_metinleri"] = []
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
    
    baglam = ""
    if st.session_state["dokuman_metinleri"]:
        try:
            tum_metinler = st.session_state["dokuman_metinleri"]
            vektorlestirici = TfidfVectorizer()
            tfidf_matrisi = vektorlestirici.fit_transform(tum_metinler)
            soru_vektoru = vektorlestirici.transform([kullanici_sorusu])
            
            benzerlikler = cosine_similarity(soru_vektoru, tfidf_matrisi).flatten()
            en_yakin_indeksler = benzerlikler.argsort()[-3:][::-1]
            
            secilen_parcalar = []
            for idx in en_yakin_indeksler:
                if benzerlikler[idx] > 0.05:
                    secilen_parcalar.append(tum_metinler[idx])
            
            if secilen_parcalar:
                baglam = "\n\n".join(secilen_parcalar)
        except Exception as e:
            pass

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
