# dash_company_dashboard.py
"""
Dash‑tabanlı şirket gösterge paneli
=================================

• Supabase'deki **company_info** ve **excel_metrics** tablolarını kullanır.
• Basit HTTP temel kimlik doğrulaması (env değişkenleriyle).
• Şirket arama çubuğu, favori yıldızlama, radar listesi.
• Her şirket için tüm grafikler dikey sıralı – kaydırılabilir.

Gerekli pip paketleri
---------------------
```bash
pip install dash dash-bootstrap-components dash-auth pandas plotly sqlalchemy psycopg2-binary
```

Ortam değişkenleri
-------------------
```
DASH_USER, DASH_PASS           # giriş bilgileri
DB_HOST, DB_NAME, DB_USER, DB_PASS, DB_PORT  # Supabase Postgres
PORT                           # Render vb. için (varsayılan 10000)
SECRET_KEY                     # Flask session anahtarı (opsiyonel)
```
"""

import os
from functools import lru_cache
from urllib.parse import parse_qs, urlparse

import dash
import dash_auth
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# ──────────────────────────────────────────────────────────────────────────────
# GİRİŞ / TEMEL KİMLİK DOĞRULAMA
# ──────────────────────────────────────────────────────────────────────────────
VALID_USERS = {
    os.getenv("DASH_USER", "user"): os.getenv("DASH_PASS", "1234")
}

external_styles = [
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
]

import secrets   # <–– en üste import ekle

app = dash.Dash(
    __name__,
    external_stylesheets=external_styles,
    suppress_callback_exceptions=True,
)

SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
app.server.config["SECRET_KEY"] = SECRET_KEY          # dash_auth bunun varlığını arıyor

server = app.server
_ = dash_auth.BasicAuth(app, VALID_USERS)  # noqa: F841 – kullanılmıyor ama gerekli

# ──────────────────────────────────────────────────────────────────────────────
# VERİ TABANI BAĞLANTISI (SQLALCHEMY)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Tek bir SQLAlchemy Engine nesnesini önbelleğe alarak tekrar kullan."""
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', 5432)}/{os.getenv('DB_NAME')}",
        pool_pre_ping=True,
    )


def load_company_info() -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql("SELECT * FROM company_info", conn)

def load_metrics() -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql("SELECT * FROM excel_metrics", conn)


# ──────────────────────────────────────────────────────────────────────────────
# SABİTLER
# ──────────────────────────────────────────────────────────────────────────────
CHART_METRICS = [
    "Fiyat",
    "Tahmin",
    "MCap/CATS",
    "Capex/Amort",
    "EBIT Margin",
    "FCF Margin",
    "Gross Margin",
    "CATS",
    "Satış Çeyrek",
    "EBIT Çeyrek",
    "Net Kar Çeyrek",
]

# ──────────────────────────────────────────────────────────────────────────────
# KULLANICI ARAYÜZÜ BİLEŞENLERİ
# ──────────────────────────────────────────────────────────────────────────────

def search_input():
    return dbc.Input(
        id="search-input",
        placeholder="Ticker ara…",
        type="text",
        debounce=True,
        style={"maxWidth": "220px"},
    )


def star_icon(filled: bool):
    return html.I(
        className="bi bi-star-fill" if filled else "bi bi-star",
        style={"fontSize": "1.8rem", "color": "#f7c948", "cursor": "pointer"},
        id="fav-toggle",
        n_clicks=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# UYGULAMA İSKELETİ
# ──────────────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="store-favs", storage_type="local"),  # kalıcı favoriler
        dcc.Store(id="store-radar", storage_type="session"),
        # Üst şerit -------------------------------------------------------------
        dbc.Row(
            [
                dbc.Col(html.H3("Şirket Dashboard", className="text-primary"), sm=6),
                dbc.Col(
                    [
                        search_input(),
                        dbc.Button("Radar", id="btn-radar", color="secondary", outline=True, className="ms-2"),
                        dbc.Button("Favoriler", id="btn-favs", color="secondary", outline=True, className="ms-1"),
                    ],
                    sm=6,
                    className="text-end",
                ),
            ],
            align="center",
            className="mt-2 mb-2",
        ),
        html.Hr(),
        html.Div(id="page-content"),
    ],
    fluid=True,
    style={"maxWidth": "820px"},
)

# ──────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────────────────────────

def get_all_tickers() -> list[str]:
    return load_company_info()["ticker"].sort_values().tolist()


def parse_ticker_from_href(href: str | None) -> str | None:
    if not href:
        return None
    qs = parse_qs(urlparse(href).query)
    return qs.get("t", [None])[0]


def make_company_link(ticker: str):
    return dcc.Link(ticker, href=f"/?t={ticker}")


# ──────────────────────────────────────────────────────────────────────────────
# CALLBACK: SEARCH BAR → YÖNLENDİRME
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(Output("url", "href"), Input("search-input", "value"), prevent_initial_call=True)
def on_search(val):
    if not val:
        raise PreventUpdate
    val_low = val.lower()
    matches = [t for t in get_all_tickers() if val_low in t.lower()]
    if not matches:
        raise PreventUpdate
    return f"/?t={matches[0]}"


# ──────────────────────────────────────────────────────────────────────────────
# CALLBACK: ANA SAYFA YÖNLENDİRME / İÇERİK
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("page-content", "children"),
    [
        Input("url", "href"),
        Input("btn-favs", "n_clicks"),
        Input("btn-radar", "n_clicks"),
    ],
    [State("store-favs", "data"), State("store-radar", "data")],
)
def render_page(href, fav_click, radar_click, favs, radar_data):
    ctx = callback_context
    trig_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "url"

    favs = favs or []
    radar_data = radar_data or []

    if trig_id == "btn-favs":
        return favorites_layout(favs)
    if trig_id == "btn-radar":
        return radar_layout(radar_data)

    ticker = parse_ticker_from_href(href) or (get_all_tickers()[0] if get_all_tickers() else None)
    return company_layout(ticker, favs) if ticker else html.Div("Şirket bulunamadı.")


# ──────────────────────────────────────────────────────────────────────────────
# ŞİRKET SAYFASI
# ──────────────────────────────────────────────────────────────────────────────

def company_layout(ticker: str, favs: list[str]):
    info_df = load_company_info()
    metrics_df = load_metrics()

    info_row = info_df[info_df.ticker == ticker]
    if info_row.empty:
        return html.Div("Geçersiz ticker.")
    info = info_row.iloc[0]
    is_fav = ticker in favs

    # Üst başlık --------------------------------------------------------------
    header = dbc.Row(
        [
            dbc.Col(html.H4(ticker, className="mb-0"), width="auto"),
            dbc.Col(star_icon(is_fav), width="auto"),
        ],
        align="center",
        className="gap-2",
    )

    # Bilgi tablosu -----------------------------------------------------------
    table = html.Table(
        [
            html.Tr([html.Th("Sector"), html.Td(info["sector"]) ]),
            html.Tr([html.Th("Industry"), html.Td(info["industry"]) ]),
            html.Tr([html.Th("Employees"), html.Td(f"{info['employees']:,}") ]),
            html.Tr([html.Th("Earnings Date"), html.Td(str(info["earnings_date"])) ]),
            html.Tr([html.Th("Market Cap"), html.Td(str(info["market_cap"])) ]),
            html.Tr([html.Th("Radar"), html.Td(str(info["radar"])) ]),
        ],
        className="table table-sm",
    )

    # Summary ---------------------------------------------------------------
    summary = html.Div(
        [html.H5("Summary"), html.Pre(info["summary"], style={"whiteSpace": "pre-wrap"})],
        className="p-2 border rounded",
    )

    # Grafikler --------------------------------------------------------------
    charts = [metric_chart(metrics_df, ticker, m) for m in CHART_METRICS]
    charts = [c for c in charts if c]
    charts_container = html.Div(charts, style={"maxHeight": "65vh", "overflowY": "auto"})

    return html.Div([header, html.Hr(), table, summary, html.Hr(), charts_container])


# ──────────────────────────────────────────────────────────────────────────────
# GRAFİK ÜRETİMİ
# ──────────────────────────────────────────────────────────────────────────────

def metric_chart(df: pd.DataFrame, ticker: str, metric: str):
    data = df[(df.ticker == ticker) & (df.metric == metric)].dropna(subset=["period", "value"])
    if data.empty:
        return None

    # "none" metinlerini eleyip sayıya çevir
    data = data[data.value.astype(str).str.lower() != "none"].copy()
    data["value"] = data.value.astype(float)
    data = data.sort_values("period")

    # "Fiyat" özel durumu: Tahmin'le birlikte çiz ---------------------------
    if metric == "Fiyat":
        tahmin = df[(df.ticker == ticker) & (df.metric == "Tahmin")].dropna(subset=["period", "value"])
        tahmin = tahmin[tahmin.value.astype(str).str.lower() != "none"].copy()
        tahmin["value"] = tahmin.value.astype(float)
        tahmin = tahmin.sort_values("period")

        if data.empty and tahmin.empty:
            return None

        fig = go.Figure()
        if not data.empty:
            fig.add_trace(
                go.Scatter(x=data.period, y=data.value, mode="lines+markers", name="Fiyat", line=dict(color="royalblue"))
            )
        if not tahmin.empty:
            fig.add_trace(
                go.Scatter(x=tahmin.period, y=tahmin.value, mode="lines+markers", name="Tahmin", line=dict(color="chocolate"))
            )
        fig.update_layout(title="Fiyat & Tahmin", margin=dict(l=10, r=10, t=30, b=20), height=260)
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    # Diğer metrikler --------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.period, y=data.value, mode="lines+markers", name=metric))
    fig.update_layout(title=metric, margin=dict(l=10, r=10, t=30, b=20), height=230)
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


# ──────────────────────────────────────────────────────────────────────────────
# FAVORİ İŞLEMLERİ
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("store-favs", "data"),
    Input("fav-toggle", "n_clicks"),
    State("url", "href"),
    State("store-favs", "data"),
    prevent_initial_call=True,
)
def toggle_fav(_, href, favs):
    ticker = parse_ticker_from_href(href)
    if not ticker:
        raise PreventUpdate
    favs = favs or []
    favs.remove(ticker) if ticker in favs else favs.append(ticker)
    return favs


def favorites_layout(favs: list[str]):
    return html.Div(
        [
            html.H4("⭐ Favoriler"),
            html.Hr(),
            html.Ul([html.Li(make_company_link(t), className="mb-2") for t in favs]) if favs else html.Div("Favorilere eklenmiş şirket yok."),
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# RADAR LİSTESİ
# ──────────────────────────────────────────────────────────────────────────────

def current_radar_list() -> list[str]:
    return load_company_info()[lambda d: d.radar == 1].ticker.tolist()


def radar_layout(radars: list[str]):
    return html.Div(
        [
            html.H4("🕵️ Radar Listesi"),
            dbc.Button("Listeyi Güncelle", id="btn-update-radar", color="info", outline=True, size="sm", className="mb-2"),
            html.Hr(),
            html.Ul([html.Li(make_company_link(t), className="mb-2") for t in radars]) if radars else html.Div("Radar'da şirket yok."),
        ]
    )


@app.callback(Output("store-radar", "data"), Input("btn-update-radar", "n_clicks"), prevent_initial_call=True)
def on_radar_update(_):
    return current_radar_list()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
