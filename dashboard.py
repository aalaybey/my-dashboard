import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth

# --- Kullanıcı bilgileri ---
users = [
    {
        "name": "Alper",
        "username": "aalaybey",
        "password": "$2a$12$jahr8psB.vh3Y6mIHccYP.zOo5C0CfYcyJBvcwxFe.ncqQSFomKkC"
    }
]

authenticator = stauth.Authenticate(
    users,
    "my_dashboard_cookie", "my_dashboard_signature_key", cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login("Giriş Yap", location="main")

if not authentication_status:
    st.warning("Giriş yapmalısınız.")
    st.stop()
elif authentication_status is None:
    st.info("Lütfen giriş bilgilerinizi girin.")
    st.stop()


DB_HOST = st.secrets["DB_HOST"]
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
DB_PORT = st.secrets["DB_PORT"]

# --- DB'DEN DATA ÇEK ---
@st.cache_data(show_spinner=False)
def load_data():
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME,
        user=DB_USER, password=DB_PASS, port=DB_PORT
    )
    df = pd.read_sql("SELECT * FROM excel_metrics", conn)
    conn.close()
    return df

st.title("Excel'den Supabase'e - Kişisel Dashboard")

# --- DATA ---
with st.spinner("Veriler yükleniyor..."):
    df = load_data()

# --- FİLTRELEME ---
tickers = df['ticker'].dropna().unique()
metrics = df['metric'].dropna().unique()

col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("Şirket (Ticker) Seç", sorted(tickers))
with col2:
    metric = st.selectbox("Metrik Seç", sorted(metrics))

periods = df.loc[(df['ticker'] == ticker) & (df['metric'] == metric), 'period']
values = df.loc[(df['ticker'] == ticker) & (df['metric'] == metric), 'value']

# --- GRAFİK ---
if len(values) == 0:
    st.warning("Bu seçimde veri bulunamadı.")
else:
    chart_df = pd.DataFrame({'Dönem': periods, 'Değer': pd.to_numeric(values, errors='coerce')})
    chart_df = chart_df.sort_values('Dönem')

    st.subheader(f"{ticker} - {metric}")
    fig, ax = plt.subplots()
    ax.plot(chart_df['Dönem'], chart_df['Değer'], marker='o')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Ham Veriler")
    st.dataframe(chart_df, use_container_width=True)
