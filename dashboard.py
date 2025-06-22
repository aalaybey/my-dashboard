# dashboard.py
import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth  # ★ yeni

# ------------------------------------------------------------------------------
# 1) KULLANICI / ŞİFRE TANIMI
#    İsterseniz bunları st.secrets içinde tutabilirsiniz; örnek basit olması için
#    kod içinde gösteriliyor.
# ------------------------------------------------------------------------------
NAMES = ["Alper"]           # Görünecek isim
USERNAMES = ["aalaybey"]       # Oturum açarken yazılacak kullanıcı adı


# Parolaları tek seferlik hash’lemek için:
#   from streamlit_authenticator import Hasher
#   hashed_pw = Hasher(["12345"]).generate()
#   print(hashed_pw)
# çıktısını aşağıdaki listeye yapıştırın → düz metin saklamamış olursunuz.
HASHED_PASSWORDS = ["$2a$12$MKw.S2MU0uKGQBzoa.vtVuqPVYlqMNJBDnquVSpZ4eoFe1LXXeFn2"]

# ------------------------------------------------------------------------------
# 2) GİRİŞ EKRANI
# ------------------------------------------------------------------------------
authenticator = stauth.Authenticate(
    NAMES,
    USERNAMES,
    HASHED_PASSWORDS,
    st.secrets["COOKIE_NAME"],
    st.secrets["SIGN_KEY"],
    cookie_expiry_days=1
)

with st.sidebar:
    st.title("🔐 Giriş Yap")
    name, auth_status, username = authenticator.login("Oturum Aç", "main")

if auth_status is False:
    st.error("❌ Kullanıcı adı veya şifre hatalı")
    st.stop()
elif auth_status is None:
    st.warning("ℹ️ Lütfen giriş bilgilerinizi girin")
    st.stop()

# ------------------------------------------------------------------------------
# 3) YETKİLİ KULLANICI ARAYÜZÜ
# ------------------------------------------------------------------------------
authenticator.logout("Çıkış", "sidebar")

st.title("Excel'den Supabase'e – Kişisel Dashboard")

# ---- Veritabanı bilgileri (st.secrets altında saklamaya devam) --------------
DB_HOST = st.secrets["DB_HOST"]
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
DB_PORT = st.secrets["DB_PORT"]

@st.cache_data(show_spinner=False)
def load_data():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
    )
    df = pd.read_sql("SELECT * FROM excel_metrics", conn)
    conn.close()
    return df

with st.spinner("🔄 Veriler yükleniyor..."):
    df = load_data()

# ---- Filtreler ---------------------------------------------------------------
tickers = df["ticker"].dropna().unique()
metrics = df["metric"].dropna().unique()

col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Şirket (Ticker) Seç", sorted(tickers))
with col2:
    metric = st.selectbox("Metrik Seç", sorted(metrics))

periods = df.loc[(df["ticker"] == ticker) & (df["metric"] == metric), "period"]
values = df.loc[(df["ticker"] == ticker) & (df["metric"] == metric), "value"]

# ---- Grafik ------------------------------------------------------------------
if len(values) == 0:
    st.warning("Bu seçimde veri bulunamadı.")
else:
    chart_df = pd.DataFrame(
        {"Dönem": periods, "Değer": pd.to_numeric(values, errors="coerce")}
    ).sort_values("Dönem")

    st.subheader(f"{ticker} – {metric}")
    fig, ax = plt.subplots()
    ax.plot(chart_df["Dönem"], chart_df["Değer"], marker="o")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Ham Veriler")
    st.dataframe(chart_df, use_container_width=True)
