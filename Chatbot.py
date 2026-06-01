from openai import OpenAI
import streamlit as st

st.set_page_config(
    page_title="Yapay Zekamıza Her Şeyi Sorun", 
    page_icon="✨", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container { padding-top: 4rem; max-width: 800px; }
    
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
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
    
    .centered-title {
        text-align: center;
        font-weight: 400;
        margin-top: 10px;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.title("⚙️ Yapılandırma")
    
    openai_api_key = st.text_input(
        "OpenAI API Anahtarı", 
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

if prompt := st.chat_input("Projeleriniz hakkında bana her şeyi sorun"):
    if not openai_api_key:
        st.info("Lütfen devam etmek için sol menüden (Sidebar) OpenAI API anahtarınızı girin.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

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
