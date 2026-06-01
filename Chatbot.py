from openai import OpenAI
import streamlit as st

# Sayfa Genişlik ve Başlık Ayarları
st.set_page_config(page_title="Gelişmiş Yapay Zeka Asistanı", page_icon="🤖", layout="wide")

# Gelişmiş Yan Menü (Sidebar) Ayarları
with st.sidebar:
    st.title("⚙️ Yapılandırma")
    
    # 1. API Anahtarı Girişi
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        key="chatbot_api_key", 
        type="password",
        placeholder="sk-..."
    )
    
    # 2. Model Seçimi (Kullanıcıya esneklik sağlama)
    model_choice = st.selectbox(
        "Kullanılacak Model",
        options=["gpt-4o", "gpt-4o-mini"],
        index=1,
        help="gpt-4o daha zeki ve yeteneklidir, gpt-4o-mini ise daha hızlı ve ekonomiktir."
    )
    
    # 3. Gelişmiş Parametre Ayarları (Sistem Karakteri)
    system_instruction = st.text_area(
        "Sistem Talimatı (System Prompt)",
        value="Sen yardımsever, profesyonel ve net cevaplar veren bir yapay zeka asistanısın.",
        help="Botun nasıl bir karakter veya uzmanlık takınacağını buradan belirleyebilirsiniz."
    )
    
    st.markdown("---")
    
    # 4. Sohbeti Temizleme Butonu
    if st.button("🔄 Sohbet Geçmişini Temizle", use_container_width=True):
        st.session_state["messages"] = [{"role": "assistant", "content": "Sohbet geçmişi sıfırlandı. Nasıl yardımcı olabilirim?"}]
        st.rerun()

    st.markdown("---")
    st.markdown("[🔑 OpenAI API Key Al](https://platform.openai.com/account/api-keys)")

# Ana Sayfa Başlığı ve Tasarımı
st.title("💬 Kurumsal Yapay Zeka Asistanı")
st.caption("🚀 Gelişmiş Streamlit & OpenAI Chatbot Altyapısı")

# Sohbet geçmişi başlatılmamışsa ilk mesajı ekle
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Merhaba! Size bugün nasıl yardımcı olabilirim?"}]

# Geçmiş mesajları ekrana bas (Daha şık ikonlar ve rollerle)
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue  # Sistem mesajını ekranda gizle
    avatar = "🤖" if msg["role"] == "assistant" else "🧑‍💻"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# Kullanıcı Giriş Alanı
if prompt := st.chat_input("Mesajınızı yazın..."):
    if not openai_api_key:
        st.info("Lütfen devam etmek için yan menüden OpenAI API anahtarınızı girin.")
        st.stop()

    # OpenAI istemcisini başlat
    client = OpenAI(api_key=openai_api_key)
    
    # Kullanıcı mesajını ekrana ve hafızaya ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    # API'ye gönderilecek güncel mesaj listesini hazırla (Sistem promptunu her zaman en başa ekle)
    api_messages = [{"role": "system", "content": system_instruction}]
    # Mevcut geçmişteki system prompt harici mesajları ekle
    api_messages.extend([m for m in st.session_state.messages if m["role"] != "system"])

    # Asistanın cevabı için bir alan (container) oluştur ve akışı başlat
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Gelişmiş Akış (Streaming) Özelliği
            stream = client.chat.completions.create(
                model=model_choice,
                messages=api_messages,
                stream=True,  # Kelime kelime ekrana yazdırmayı açar
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    # Yazı efekti için anlık güncelleme yapıyoruz
                    message_placeholder.markdown(full_response + "▌")
            
            # Akış bittiğinde imleci kaldır ve son metni sabitle
            message_placeholder.markdown(full_response)
            
            # Asistanın cevabını hafızaya kaydet
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
