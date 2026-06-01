from openai import OpenAI
import streamlit as st

# ==============================
# 📊 1. VERİ SETİ (CUSTOM DATASET)
# ==============================

veri_seti = [
    {"kategori": "proje", "anahtar": ["proje", "ödev", "assignment"]},
    {"kategori": "analiz", "anahtar": ["performans", "analiz", "veri"]},
    {"kategori": "yardım", "anahtar": ["yardım", "nasıl", "help"]},
    {"kategori": "teknik", "anahtar": ["kod", "hata", "bug"]},
]

# ==============================
# 🔧 2. VERİ İŞLEME (PREPROCESSING)
# ==============================

def veri_temizle(metin):
    metin = metin.lower()
    metin = metin.strip()
    return metin

def anahtar_kelime_bul(metin):
    kelimeler = metin.split()
    return kelimeler

# ==============================
# 🧠 3. KENDİ MODELİN (RULE-BASED MODEL)
# ==============================

def kategori_tahmin(metin):
    metin = veri_temizle(metin)
    kelimeler = anahtar_kelime_bul(metin)

    skorlar = {}

    for veri in veri_seti:
        kategori = veri["kategori"]
        skor = 0

        for kelime in kelimeler:
            if kelime in veri["anahtar"]:
                skor += 1

        skorlar[kategori] = skor

    # en yüksek skoru seç
    en_iyi = max(skorlar, key=skorlar.get)

    return en_iyi

# ==============================
# 🎯 4. ALGORİTMA KARAR MEKANİZMASI
# ==============================

def sistem_mesaji_uret(kategori):
    if kategori == "proje":
        return "Kullanıcının proje geliştirmesine yardımcı olan bir uzmansın."
    elif kategori == "analiz":
        return "Veri analizi yapan bir yapay zekasın."
    elif kategori == "yardım":
        return "Kullanıcıya adım adım açıklayan bir asistansın."
    elif kategori == "teknik":
        return "Kod hatalarını çözen teknik bir uzmansın."
    else:
        return "Genel amaçlı yardımcı bir yapay zekasın."

# ==============================
# 🎨 UI
# ==============================

st.set_page_config(page_title="Yapay Zeka Projesi", page_icon="✨")

st.title("✨ Gelişmiş Yapay Zeka Asistanı")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.header("Ayarlar")
    api_key = st.text_input("OpenAI API Key", type="password")

# ==============================
# 💬 CHAT
# ==============================

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Sorunu yaz..."):

    if not api_key:
        st.warning("API key gir")
        st.stop()

    # Kullanıcı mesajı ekle
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ==============================
    # 🔥 MODELİN ÇALIŞTIĞI YER
    # ==============================

    kategori = kategori_tahmin(prompt)
    sistem_mesaji = sistem_mesaji_uret(kategori)

    client = OpenAI(api_key=api_key)

    messages = [{"role": "system", "content": sistem_mesaji}] + st.session_state.messages

    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        cevap = response.choices[0].message.content
        st.write(cevap)

    st.session_state.messages.append({"role": "assistant", "content": cevap})
