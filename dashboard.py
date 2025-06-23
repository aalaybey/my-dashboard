# dashboard.py
import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth  # ★ yeni

# ───────── 1) KULLANICILAR ─────────
NAMES       = ["Alper"]
USERNAMES   = st.secrets["USERNAMES"]
HASHED_PWS  = st.secrets["HASHED_PWS"]

credentials = {
    "usernames": {
        u: {"name": n, "password": pw}
        for u, n, pw in zip(USERNAMES, NAMES, HASHED_PWS)
    }
}

authenticator = stauth.Authenticate(
    credentials,
    st.secrets["COOKIE_NAME"],
    st.secrets["SIGN_KEY"],
    cookie_expiry_days=1,
)

# ───────── 2) GİRİŞ FORMU ─────────
authenticator.login(
    "main",
    fields={
        "Form name": "Oturum Aç",
        "Login":     "Giriş",
        "Username":  "Kullanıcı adı",
        "Password":  "Şifre",
    },
)
auth_status = st.session_state.get("authentication_status")

# ───────── 3) DURUM KONTROLÜ ─────────
if auth_status is False:
    st.error("❌ Kullanıcı adı veya şifre hatalı")
    st.stop()

elif auth_status is None:
    st.stop()                     # ⚠️ Uyarı gösterme, sessizce bekle

authenticator.logout("Çıkış", "main")

st.title("Excel'den Supabase'e – Kişisel Dashboard")

# ---- Veritabanı bilgileri (st.secrets altında saklamaya devam) --------------
DB_HOST = st.secrets["DB_HOST"]
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
DB_PORT = st.secrets["DB_PORT"]

@st.cache_data(ttl=120, show_spinner=False)   # 120 saniyede bir otomatik tazele
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

# --- Kullanıcı manuel yenilemek isterse ---
if st.button("🔄 Verileri yenile"):
    st.cache_data.clear()   # önbelleği temizle
    st.rerun()              # sayfayı baştan çalıştır


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
        {"Dönem": periods, "Değer": values}  # 1) ham çerçeve
    )

    # 2) Eksik satırları at, tipleri düzelt
    chart_df = chart_df.dropna(subset=["Dönem", "Değer"])
    chart_df["Dönem"] = chart_df["Dönem"].astype(str)  # x-ekseni string
    chart_df["Değer"] = pd.to_numeric(chart_df["Değer"], errors="coerce")

    # 3) Son temizlik: yine boş kaldıysa uyar ve dur
    if chart_df.empty:
        st.warning("Bu metrik için gösterilecek veri yok.")
        st.stop()

    chart_df = chart_df.sort_values("Dönem")  # 4) sıralayıp devam

    st.subheader(f"{ticker} – {metric}")
    fig, ax = plt.subplots()
    ax.plot(chart_df["Dönem"], chart_df["Değer"], marker="o")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Ham Veriler")
    st.dataframe(chart_df, use_container_width=True)
