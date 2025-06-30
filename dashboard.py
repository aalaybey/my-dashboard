# app.py
import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from collections import defaultdict
import streamlit_authenticator as stauth

st.set_page_config(layout="wide", page_title="Şirket Dashboard")

# ────────── KULLANICI ve DB AYARLARI ──────────
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
    cookie_expiry_days=30,   # daha uzun süreli hatırlama için süreyi uzatabilirsin
    cookie_secure=True,      # HTTPS kullanıyorsan bunu True yap, HTTP kullanıyorsan False yap
)

name, auth_status, username = authenticator.login(
    "main",
    fields={
        "Form name": "Oturum Aç",
        "Login":     "Giriş",
        "Username":  "Kullanıcı adı",
        "Password":  "Şifre",
    },
    # ÖNEMLİ: Tekden çok sekme/cihaz desteği için single_session=False!
    single_session=False,
)

if auth_status is False:
    st.error("❌ Kullanıcı adı veya şifre hatalı")
    st.stop()
elif auth_status is None:
    st.stop()
# Oturum açıkken çıkış butonu
authenticator.logout("Çıkış", "main")

DB_HOST = st.secrets["DB_HOST"]
DB_NAME = st.secrets["DB_NAME"]
DB_USER = st.secrets["DB_USER"]
DB_PASS = st.secrets["DB_PASS"]
DB_PORT = st.secrets["DB_PORT"]

@st.cache_data(ttl=120)
def load_metrics():
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

@st.cache_data(ttl=120)
def load_company_info():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
    )
    df = pd.read_sql("SELECT * FROM company_info", conn)
    conn.close()
    return df

# ────────── VERİLERİ ÇEK ──────────
with st.spinner("🔄 Veriler yükleniyor..."):
    metrics_df = load_metrics()
    info_df = load_company_info()

all_tickers = sorted(info_df['ticker'].unique())

# ────────── FAVORİLER / RADAR STATE ──────────
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = set()
if 'selected_ticker' not in st.session_state:
    st.session_state['selected_ticker'] = all_tickers[0] if all_tickers else None

def set_fav(ticker):
    if ticker in st.session_state['favorites']:
        st.session_state['favorites'].remove(ticker)
    else:
        st.session_state['favorites'].add(ticker)

def go_ticker(ticker):
    st.session_state['selected_ticker'] = ticker

# ────────── SEARCH BAR (NAVBAR YERİNE) ──────────
def searchbar():
    st.markdown(
        """
        <style>
        .stTextInput > div > div > input {
            font-size: 22px !important;
            padding: 12px 8px;
        }
        @media (max-width:600px){
            .stTextInput > div > div > input {
                font-size: 18px !important;
                padding: 14px 6px;
            }
        }
        </style>
        """, unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns([2, 4, 1, 1])

    # Arama kutusu
    with c1:
        search = st.text_input("Şirket Ara", "", key="searchbar", placeholder="Şirket ticker yaz...", label_visibility="collapsed")
    # Filter tickers (case-insensitive, substring)
    search_lower = search.strip().lower()
    matched_tickers = [t for t in all_tickers if search_lower in t.lower()] if search_lower else all_tickers

    if matched_tickers:
        # Seçili ticker matched listede değilse, otomatik ilkine geç
        if st.session_state['selected_ticker'] not in matched_tickers:
            st.session_state['selected_ticker'] = matched_tickers[0]
        sel = st.selectbox(
            "Şirket Seç", matched_tickers,
            index=matched_tickers.index(st.session_state['selected_ticker']),
            key="select_ticker", label_visibility="collapsed"
        )
        if sel != st.session_state['selected_ticker']:
            go_ticker(sel)
            st.rerun()
    else:
        st.warning("Hiçbir şirket bulunamadı.")

    # Fav, yenile
    with c2:
        ticker = st.session_state['selected_ticker']
        is_fav = ticker in st.session_state['favorites']
        star = "★" if is_fav else "☆"
        col_star, col_refresh = st.columns([6, 1], gap="small")
        with col_star:
            if st.button(star, help="Favorilere ekle/çıkar", key=f"fav_{ticker}", use_container_width=True):
                set_fav(ticker)
        with col_refresh:
            if st.button("🔄", help="Verileri Yenile", key=f"refresh_{ticker}", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
    # Radar
    with c3:
        if st.button("Radar"):
            st.session_state['nav'] = "radar"
            st.rerun()
    # Favoriler
    with c4:
        if st.button("Favoriler"):
            st.session_state['nav'] = "favorites"
            st.rerun()

# NAV State: company, radar, favorites
if 'nav' not in st.session_state:
    st.session_state['nav'] = 'company'

# Yardımcı: float dönüşümünde hata olursa None dön
def tofloat(x):
    try:
        if x is None: return None
        if pd.isna(x): return None
        if isinstance(x, str) and x.lower() in ('none', ''): return None
        return float(x)
    except:
        return None

# ────────── MAIN BODY ──────────
def company_page(ticker):
    searchbar()
    info = info_df[info_df['ticker'] == ticker].iloc[0]
    st.markdown(f"### {ticker}  ")
    # Company Info Card
    infoc1, infoc2 = st.columns([2, 3])
    with infoc1:
        st.markdown(f"""
        <table style="font-size:16px;">
        <tr><td><b>Sector</b></td><td>{info['sector']}</td></tr>
        <tr><td><b>Industry</b></td><td>{info['industry']}</td></tr>
        <tr><td><b>Employees</b></td><td>{info['employees']:,}</td></tr>
        <tr><td><b>Earnings Date</b></td><td>{info['earnings_date']}</td></tr>
        </table>
        """, unsafe_allow_html=True)
    with infoc2:
        st.markdown("#### Summary")
        st.markdown(f"<div style='font-size:13px'>{info['summary']}</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Grafiksel metrikler
    metrics_for_chart = [
        "Fiyat", "Tahmin", "MCap/CATS", "Capex/Amort", "EBIT Margin", "FCF Margin",
        "Gross Margin", "CATS", "Satış Çeyrek", "EBIT Çeyrek", "Net Kar Çeyrek"
    ]
    df = metrics_df[metrics_df['ticker'] == ticker]
    metric_data = {m: df[df['metric'] == m].sort_values('period') for m in metrics_for_chart}

    # Fiyat & Tahmin birlikte çizgi grafiği
    fiyat = metric_data["Fiyat"].dropna(subset=['period', 'value'])
    fiyat = fiyat[(fiyat['value'].notnull()) & (fiyat['period'].notnull())]
    fiyat = fiyat[fiyat['value'] != 'None']
    fiyat = fiyat[fiyat['period'] != 'None']

    tahmin = metric_data["Tahmin"].dropna(subset=['period', 'value'])
    tahmin = tahmin[(tahmin['value'].notnull()) & (tahmin['period'].notnull())]
    tahmin = tahmin[tahmin['value'] != 'None']
    tahmin = tahmin[tahmin['period'] != 'None']

    if not fiyat.empty or not tahmin.empty:
        st.markdown("#### Fiyat & Tahmin")
        fig, ax = plt.subplots(figsize=(8, 4))

        # --- BURADA DÖNÜŞÜMLERİ EKLE ---
        if not fiyat.empty:
            fiyat = fiyat.copy()
            fiyat['value'] = fiyat['value'].astype(float)
            fiyat = fiyat.sort_values('period')   # period sırası önemli
            ax.plot(fiyat['period'], fiyat['value'], marker='o', label="Fiyat", color='royalblue')

        if not tahmin.empty:
            tahmin = tahmin.copy()
            tahmin['value'] = tahmin['value'].astype(float)
            tahmin = tahmin.sort_values('period')
            ax.plot(tahmin['period'], tahmin['value'], marker='o', label="Tahmin", color='chocolate')

        ax.set_ylabel("Değer")
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    # Diğer metrikler
    for m in metrics_for_chart:
        if m in ["Fiyat", "Tahmin"]: continue
        d = metric_data[m].dropna(subset=['period', 'value'])
        d = d[(d['value'].notnull()) & (d['period'].notnull())]
        d = d[d['value'] != 'None']
        d = d[d['period'] != 'None']
        if not d.empty:
            d = d.copy()
            d['value'] = d['value'].astype(float)
            d = d.sort_values('period')
            st.markdown(f"#### {m}")
            fig, ax = plt.subplots()
            ax.plot(d['period'], d['value'], marker='o')
            ax.set_ylabel("Değer")
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    # Alt: Ham metrik tablo
    st.markdown("#### Tüm Ham Veriler")
    st.dataframe(df.pivot(index="period", columns="metric", values="value").sort_index(), use_container_width=True)

# ────────── FAVORİLER SAYFASI ──────────
def favorites_page():
    searchbar()
    favs = list(st.session_state['favorites'])
    st.markdown("## ⭐ Favori Şirketler")
    if not favs:
        st.info("Hiç favori şirket eklemedin.")
    else:
        for t in favs:
            st.markdown(f"- [{t}](?selected_ticker={t})", unsafe_allow_html=True)

# ────────── RADAR SAYFASI ──────────
def radar_page():
    # --- STATE: radar şirketleri ---
    if "radar_list" not in st.session_state:
        st.session_state["radar_list"] = []

    # --- Listeyi güncelleyen fonksiyon ---
    def calculate_radar_list():
        df = metrics_df.copy()
        import re
        def period_to_sortable(period):
            if pd.isna(period):
                return (0, 0)
            s = str(period)
            m = re.search(r'(20\d{2}).*?Q([1-4])', s)
            if m:
                return (int(m.group(1)), int(m.group(2)))
            m = re.search(r'Q([1-4]).*?(20\d{2})', s)
            if m:
                return (int(m.group(2)), int(m.group(1)))
            return (0, 0)

        def get_last_forecast_and_prev_price(df, ticker):
            d_tahmin = df[(df['ticker'] == ticker) & (df['metric'] == "Tahmin")].copy()
            d_fiyat = df[(df['ticker'] == ticker) & (df['metric'] == "Fiyat")].copy()
            if d_tahmin.empty or d_fiyat.empty:
                return None, None
            d_tahmin['period_sort'] = d_tahmin['period'].apply(period_to_sortable)
            d_tahmin = d_tahmin.sort_values('period_sort')
            d_fiyat['period_sort'] = d_fiyat['period'].apply(period_to_sortable)
            d_fiyat = d_fiyat.sort_values('period_sort')
            last_forecast_row = d_tahmin.iloc[-1]
            last_period = last_forecast_row['period']
            last_value = tofloat(last_forecast_row['value'])
            fiyat_periods_sortable = list(d_fiyat['period_sort'])
            last_sort = period_to_sortable(last_period)
            prev_idx = None
            for i in range(len(fiyat_periods_sortable)):
                if fiyat_periods_sortable[i] < last_sort:
                    prev_idx = i
            if prev_idx is None:
                return last_value, None
            prev_fiyat_value = tofloat(d_fiyat.iloc[prev_idx]['value'])
            return last_value, prev_fiyat_value

        radar_list = []
        for t in all_tickers:
            tahmin_cur, fiyat_prev = get_last_forecast_and_prev_price(df, t)
            if tahmin_cur is None or fiyat_prev is None or tahmin_cur <= fiyat_prev: continue

            def get_latest(metric):
                d = df[(df['ticker'] == t) & (df['metric'] == metric)].sort_values('period')
                vals = d['value'].dropna().values
                if len(vals) == 0:
                    return None, None
                cur = tofloat(vals[-1])
                prev = tofloat(vals[-2]) if len(vals) > 1 else None
                return cur, prev

            tahmin_cur_, tahmin_prev = get_latest("Tahmin")
            mcap_cats_cur, mcap_cats_prev = get_latest("MCap/CATS")
            ebit_margin_cur, ebit_margin_prev = get_latest("EBIT Margin")
            fcf_margin_cur, fcf_margin_prev = get_latest("FCF Margin")
            cats_cur, cats_prev = get_latest("CATS")
            capex_amort_cur, capex_amort_prev = get_latest("Capex/Amort")

            if tahmin_prev is None or tahmin_cur_ is None or tahmin_cur_ <= tahmin_prev: continue
            if mcap_cats_cur is None or mcap_cats_cur <= 0: continue
            if mcap_cats_prev is not None and mcap_cats_prev > 0 and not (mcap_cats_cur < mcap_cats_prev): continue
            if ebit_margin_cur is None or ebit_margin_prev is None or ebit_margin_cur <= ebit_margin_prev: continue
            if fcf_margin_cur is None or fcf_margin_prev is None or fcf_margin_cur <= fcf_margin_prev: continue
            if cats_cur is None or cats_prev is None or cats_cur <= cats_prev: continue
            if capex_amort_cur is None or capex_amort_prev is None or capex_amort_cur >= capex_amort_prev: continue
            radar_list.append(t)
        return radar_list

    # --- GÜNCELLE BUTONU ve RADAR ŞİRKETLERİ LİSTESİ ---
    c1, c2 = st.columns([8, 1])
    with c1:
        st.markdown("### Radar Şirketleri")
        if st.session_state["radar_list"]:
            for t in st.session_state["radar_list"]:
                st.write(t)
        else:
            st.info("Radar kriterlerini sağlayan şirket yok.")
    with c2:
        if st.button("Listeyi Güncelle"):
            st.session_state["radar_list"] = calculate_radar_list()
            st.rerun()

    # --- SAYFA YÜKLENİRKEN RADAR LİSTESİ BOŞSA OLUŞTUR ---
    if not st.session_state["radar_list"]:
        st.session_state["radar_list"] = calculate_radar_list()

    # --- SEARCHBAR & ŞİRKET SEÇİMİ ---
    searchbar()

    # --- SEÇİLEN ŞİRKETİN DETAYLARI (company_page içeriği) ---
    ticker = st.session_state['selected_ticker']
    info = info_df[info_df['ticker'] == ticker].iloc[0]
    st.markdown(f"### {ticker}  ")
    infoc1, infoc2 = st.columns([2, 3])
    with infoc1:
        st.markdown(f"""
        <table style="font-size:16px;">
        <tr><td><b>Sector</b></td><td>{info['sector']}</td></tr>
        <tr><td><b>Industry</b></td><td>{info['industry']}</td></tr>
        <tr><td><b>Employees</b></td><td>{info['employees']:,}</td></tr>
        <tr><td><b>Earnings Date</b></td><td>{info['earnings_date']}</td></tr>
        </table>
        """, unsafe_allow_html=True)
    with infoc2:
        st.markdown("#### Summary")
        st.markdown(f"<div style='font-size:13px'>{info['summary']}</div>", unsafe_allow_html=True)
    st.markdown("---")

    metrics_for_chart = [
        "Fiyat", "Tahmin", "MCap/CATS", "Capex/Amort", "EBIT Margin", "FCF Margin",
        "Gross Margin", "CATS", "Satış Çeyrek", "EBIT Çeyrek", "Net Kar Çeyrek"
    ]
    df = metrics_df[metrics_df['ticker'] == ticker]
    metric_data = {m: df[df['metric'] == m].sort_values('period') for m in metrics_for_chart}

    # Fiyat & Tahmin birlikte çizgi grafiği
    fiyat = metric_data["Fiyat"].dropna(subset=['period', 'value'])
    fiyat = fiyat[(fiyat['value'].notnull()) & (fiyat['period'].notnull())]
    fiyat = fiyat[fiyat['value'] != 'None']
    fiyat = fiyat[fiyat['period'] != 'None']

    tahmin = metric_data["Tahmin"].dropna(subset=['period', 'value'])
    tahmin = tahmin[(tahmin['value'].notnull()) & (tahmin['period'].notnull())]
    tahmin = tahmin[tahmin['value'] != 'None']
    tahmin = tahmin[tahmin['period'] != 'None']

    if not fiyat.empty or not tahmin.empty:
        st.markdown("#### Fiyat & Tahmin")
        fig, ax = plt.subplots(figsize=(8, 4))
        if not fiyat.empty:
            fiyat = fiyat.copy()
            fiyat['value'] = fiyat['value'].astype(float)
            fiyat = fiyat.sort_values('period')
            ax.plot(fiyat['period'], fiyat['value'], marker='o', label="Fiyat", color='royalblue')
        if not tahmin.empty:
            tahmin = tahmin.copy()
            tahmin['value'] = tahmin['value'].astype(float)
            tahmin = tahmin.sort_values('period')
            ax.plot(tahmin['period'], tahmin['value'], marker='o', label="Tahmin", color='chocolate')
        ax.set_ylabel("Değer")
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    for m in metrics_for_chart:
        if m in ["Fiyat", "Tahmin"]: continue
        d = metric_data[m].dropna(subset=['period', 'value'])
        d = d[(d['value'].notnull()) & (d['period'].notnull())]
        d = d[d['value'] != 'None']
        d = d[d['period'] != 'None']
        if not d.empty:
            d = d.copy()
            d['value'] = d['value'].astype(float)
            d = d.sort_values('period')
            st.markdown(f"#### {m}")
            fig, ax = plt.subplots()
            ax.plot(d['period'], d['value'], marker='o')
            ax.set_ylabel("Değer")
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("#### Tüm Ham Veriler")
    st.dataframe(df.pivot(index="period", columns="metric", values="value").sort_index(), use_container_width=True)


# --- Routing ---
if st.session_state['nav'] == "company":
    company_page(st.session_state['selected_ticker'])
elif st.session_state['nav'] == "favorites":
    favorites_page()
    st.session_state['nav'] = "company"
elif st.session_state['nav'] == "radar":
    radar_page()
else:
    company_page(st.session_state['selected_ticker'])
