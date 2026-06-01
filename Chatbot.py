from openai import OpenAI
import streamlit as st

# Sayfa Ayarları
st.set_page_config(page_title="Ask Our AI Everything", page_icon="✨", layout="centered")

# Tüm temalarla uyumlu CSS + Otomatik gelen sayfa menüsünü gizleme kodu
st.markdown("""
    <style>
    /* Üst boşlukları ve genişliği optimize et */
    .block-container { padding-top: 4rem; max-width: 800px; }
    
    /* ---- CRITICAL: Streamlit'in otomatik oluşturduğu sol menüyü gizler ---- */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Öneri butonlarını mevcut temaya (Açık/Koyu) otomatik uydur */
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
    
    /* Butona hover olunca temanın ana rengine göre hafifçe parla */
    div.stButton > button:hover {
        background-color: rgba(128, 128, 128, 0.15);
        border-color: rgba(128, 128, 128, 0.3);
    }
    
    /* Başlıkların ortalanması */
    .centered-title {
        text-align: center;
        font-weight: 400;
        margin-top: 10px;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# Hafıza (Session State) Tanımlamaları
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- YAN MENÜ (SIDEBAR) ---
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
    if st.button("🔄 Sohbeti Sıfırla", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# --- ANA EKRAN TASARIMI ---
if len(st.session_state["messages"]) == 0:
    st.markdown("<h1 style='text-align: center; font-size: 45px; margin-bottom: 0;'>✨</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='centered-title'>Ask our AI anything</h2>", unsafe_allow_html=True)
    st.markdown("<p style='opacity: 0.7; font-size: 14px; margin-bottom: 12px;'>Suggestions on what to ask Our AI</p>", unsafe_allow_html=True)
    
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

else:
    for msg in st.session_state.messages:
        avatar = "✨" if msg["role"] == "assistant" else "🧑‍💻"
        st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# --- SOHBET GİRİŞ ALANI ---
if prompt := st.chat_input("Ask me anything about your projects"):
    if not openai_api_key:
        st.info("Lütfen devam etmek için sol menüden (Sidebar) OpenAI API anahtarınızı girin.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# API Yanıt Mekanizması
if len(st.session_state["messages"]) > 0 and st.session_state["messages"][-1]["role"] == "user":
    client = OpenAI(api_key=openai_api_key)
    
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
