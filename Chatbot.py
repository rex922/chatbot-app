from openai import OpenAI
import streamlit as st

# Sayfa Ayarları
st.set_page_config(page_title="Ask Our AI Everything", page_icon="✨", layout="centered")

# Görseldeki minimalist tasarımı yakalamak için özel CSS
st.markdown("""
    <style>
    /* Üst boşlukları ve gereksiz Streamlit elementlerini gizleme/düzenleme */
    .block-container { padding-top: 5rem; max-width: 800px; }
    
    /* Öneri butonlarının stilini özelleştirme */
    div.stButton > button {
        background-color: #f0f2f6;
        color: #31333F;
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 10px 15px;
        font-size: 14px;
        transition: all 0.3s ease;
        text-align: left;
        width: 100%;
        min-height: 60px;
    }
    div.stButton > button:hover {
        background-color: #e4e7eb;
        border-color: #cbd5e1;
        color: #1e293b;
    }
    </style>
""", unsafe_style=True)

# Hafıza (Session State) Tanımlamaları
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Yan Menü (Sidebar) - API Anahtarı Girişi
with st.sidebar:
    st.title("⚙️ Yapılandırma")
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        key="chatbot_api_key", 
        type="password",
        placeholder="sk-..."
    )
    
    model_choice = st.selectbox(
        "Model Seçimi",
        options=["gpt-4o", "gpt-4o-mini"],
        index=1
    )
    
    st.markdown("---")
    if st.button("🔄 Sohbeti Sıfırla"):
        st.session_state["messages"] = []
        st.rerun()

# --- ANA EKRAN TASARIMI ---

# Eğer henüz hiç mesaj yazılmadıysa görseldeki Karşılama Ekranını göster
if len(st.session_state["messages"]) == 0:
    # Üst Logo ve Başlık (Görseldeki gibi ortalanmış)
    st.markdown("<h1 style='text-align: center; font-size: 45px; margin-bottom: 0;'>✨</h1>", unsafe_style=True)
    st.markdown("<h2 style='text-align: center; font-weight: 400; color: #222; margin-top: 10px; margin-bottom: 50px;'>Ask our AI anything</h2>", unsafe_style=True)
    
    st.markdown("<p style='color: #666; font-size: 14px; margin-bottom: 10px;'>Suggestions on what to ask Our AI</p>", unsafe_style=True)
    
    # Görseldeki 3 adet öneri butonu (Yan yana sütunlar halinde)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("What can I ask you to do?"):
            st.session_state["messages"].append({"role": "user", "content": "What can I ask you to do?"})
            st.rerun()
            
    with col2:
        if st.button("Which one of my projects is performing the best?"):
            st.session_state["messages"].append({"role": "user", "content": "Which one of my projects is performing the best?"})
            st.rerun()
            
    with col3:
        if st.button("What projects should I be concerned about right now?"):
            st.session_state["messages"].append({"role": "user", "content": "What projects should I be concerned about right now?"})
            st.rerun()

# Eğer mesaj geçmişi varsa eski mesajları ekrana dök (Karşılama ekranı kaybolur)
else:
    for msg in st.session_state.messages:
        avatar = "✨" if msg["role"] == "assistant" else "🧑‍💻"
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# --- SOHBET GİRİŞ ALANI (Chat Input) ---
# Görseldeki "Ask me anything about your projects" placeholder'ı ile alt giriş alanı
if prompt := st.chat_input("Ask me anything about your projects"):
    
    # API Anahtarı Kontrolü
    if not openai_api_key:
        st.info("Lütfen devam etmek için sol menüden (Sidebar) OpenAI API anahtarınızı girin.")
        st.stop()
        
    # Eğer ilk mesaj öneri butonlarından gelmediyse, normal girdiyi ekle
    if len(st.session_state["messages"]) == 0 or st.session_state["messages"][-1]["content"] != prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# API Yanıt Tetikleme Mekanizması
if len(st.session_state["messages"]) > 0 and st.session_state["messages"][-1]["role"] == "user":
    client = OpenAI(api_key=openai_api_key)
    
    # Ekranı en son mesaja göre güncellemek için yeniden çiziyoruz
    st.rerun() if len(st.session_state["messages"]) == 1 else None 

    # Asistan yanıt alanı
    with st.chat_message("assistant", avatar="✨"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.chat.completions.create(
                model=model_choice,
                messages=st.session_state.messages,
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
