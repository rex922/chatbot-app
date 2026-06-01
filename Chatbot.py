from openai import OpenAI
import streamlit as st

# 🔥 Sayfa ayarı
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# 🎨 CSS TASARIM
st.markdown("""
<style>
/* Genel arka plan */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Chat balonları */
.stChatMessage {
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
    max-width: 70%;
}

/* Kullanıcı mesajı */
.stChatMessage[data-testid="stChatMessage-user"] {
    background-color: #2563eb;
    margin-left: auto;
}

/* Bot mesajı */
.stChatMessage[data-testid="stChatMessage-assistant"] {
    background-color: #374151;
    margin-right: auto;
}

/* Input kutusu */
textarea {
    border-radius: 10px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)

# 🔑 Sidebar
with st.sidebar:
    st.title("⚙️ Ayarlar")
    openai_api_key = st.text_input("API Key", type="password")

# 🧠 Başlık
st.title("🤖 AI Chatbot")
st.caption("Modern UI + OpenAI destekli")

# 💬 Mesaj geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Sana nasıl yardımcı olabilirim?"}
    ]

# 💬 Eski mesajları göster
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✍️ Kullanıcı input
if prompt := st.chat_input("Mesaj yaz..."):

    if not openai_api_key:
        st.warning("API key girmen lazım ⚠️")
        st.stop()

    client = OpenAI(api_key=openai_api_key)

    # kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 🤖 AI cevabı
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
