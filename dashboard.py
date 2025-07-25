# dash_company_dashboard.py
import os
import functools
import secrets
from urllib.parse import parse_qs, urlparse

import dash
import dash_auth
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from sqlalchemy import create_engine, text as sa_text
import boto3
from io import BytesIO
from botocore.client import Config  # EKLENDİ

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET")
AWS_REGION = os.getenv("AWS_REGION") or "eu-central-1"
S3_BUCKET = "alaybey"

ENDPOINT_URL = f"https://s3.{AWS_REGION}.wasabisys.com"  # EKLENDİ

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
    endpoint_url=ENDPOINT_URL,               # EKLENDİ
    config=Config(signature_version="s3v4"),  # EKLENDİ (Wasabi için zorunlu)
)

def s3_upload_text(key, text):
    s3_client.put_object(Bucket=S3_BUCKET, Key=key.strip(), Body=text.encode("utf-8"))

# ────────────── KİMLİK DOĞRULAMA ──────────────
VALID_USERS = {
    os.getenv("DASH_USER", "user"): os.getenv("DASH_PASS", "1234")
}
external_styles = [
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
]
app = dash.Dash(
    __name__,
    external_stylesheets=external_styles,
    suppress_callback_exceptions=True,
)
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
app.server.config["SECRET_KEY"] = SECRET_KEY
server = app.server
_ = dash_auth.BasicAuth(app, VALID_USERS)

# ────────────── DB ──────────────
@functools.lru_cache(maxsize=1)
def get_engine():
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', 5432)}/{os.getenv('DB_NAME')}",
        pool_pre_ping=True,
    )
def get_all_tickers():
    q = sa_text("SELECT ticker FROM company_info ORDER BY ticker")
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    return df["ticker"].tolist()

@functools.lru_cache(maxsize=128)
def load_company_info(ticker):
    q = sa_text("SELECT * FROM company_info WHERE ticker = :t")
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"t": ticker})
    return df.iloc[0] if not df.empty else None

@functools.lru_cache(maxsize=128)
def load_metrics(ticker):
    q = sa_text("""
        SELECT
            period,
            metric,
            value::numeric AS value
        FROM excel_metrics
        WHERE ticker = :t
          AND metric IN (
              'Fiyat','Tahmin','MCap/CATS','Capex/Amort','EBIT Margin',
              'FCF Margin','Gross Margin','CATS',
              'Satış Çeyrek','EBIT Çeyrek','Net Kar Çeyrek'
          )
        ORDER BY period
    """)
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn, params={"t": ticker})
    return df

def current_radar_list():
    q = sa_text("SELECT ticker FROM company_info WHERE radar = 1 ORDER BY ticker")
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn)
    return df["ticker"].tolist()

# ────────────── BİLEŞENLER ──────────────
def search_input():
    return dbc.Input(
        id="search-input",
        placeholder="Ticker ara…",
        type="text",
        debounce=True,
        style={"maxWidth": "220px"},
    )
def star_icon(filled):
    return html.I(
        className="bi bi-star-fill" if filled else "bi bi-star",
        style={"fontSize": "1.8rem", "color": "#f7c948", "cursor": "pointer"},
        id="fav-toggle",
        title="Favorilere ekle/çıkar"
    )
def make_company_link(ticker):
    return dcc.Link(ticker, href=f"/?t={ticker}")

# ────────────── LAYOUT ──────────────
app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="store-favs", storage_type="local"),
        dcc.Store(id="store-radar", storage_type="session"),
        dbc.Row(
            [
                dbc.Col(
                    search_input(),
                    sm=6,
                ),
                dbc.Col(
                    [
                        dbc.Button("Ana Sayfa", href="https://alaybey.onrender.com/", color="primary", outline=True,
                                   className="ms-1 mb-2", id="btn-home"),
                        dbc.Button("Radar", id="btn-radar", color="secondary", outline=True, className="ms-2 mb-2"),
                        dbc.Button("Favoriler", id="btn-favs", color="secondary", outline=True, className="ms-1 mb-2"),
                        # Finansal İndir butonu ana sayfadan kaldırıldı
                        dbc.Button("Verileri Güncelle", id="btn-refresh-data", color="warning", outline=False,
                                   className="ms-1 mb-2"),
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

# ────────────── YARDIMCI ──────────────
def parse_ticker_from_href(href):
    if not href:
        return None
    qs = parse_qs(urlparse(href).query)
    return qs.get("t", [None])[0]

# ────────────── SEARCH BAR CALLBACK ──────────────
@app.callback(Output("url", "href"), Input("search-input", "value"), prevent_initial_call=True)
def on_search(val):
    if not val:
        raise PreventUpdate
    val_low = val.lower()
    matches = [t for t in get_all_tickers() if val_low in t.lower()]
    if not matches:
        raise PreventUpdate
    return f"/?t={matches[0]}"

# ────────────── ANA İÇERİK CALLBACK ──────────────
@app.callback(
    Output("page-content", "children"),
    [Input("url", "href"), Input("btn-favs", "n_clicks"), Input("btn-radar", "n_clicks")],
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
    ticker = parse_ticker_from_href(href)
    if not ticker:
        # Hiçbir şirket seçili değilse sadece boş bir div döndür (veya dilersen hoş geldin mesajı ekle)
        return html.Div()
    return company_layout(ticker, favs)

# ────────────── ŞİRKET SAYFASI ──────────────
def company_layout(ticker, favs):
    info = load_company_info(ticker)
    if info is None:
        return html.Div("Geçersiz ticker.")
    metrics_df = load_metrics(ticker)
    is_fav = ticker in favs

    header = dbc.Row(
        [
            dbc.Col(html.H4(ticker, className="mb-0"), width="auto"),
            dbc.Col(star_icon(is_fav), width="auto"),
        ],
        align="center",
        className="gap-2",
    )

    def td_safe(val, binlik=False):
        if pd.isnull(val): return "-"
        if binlik:
            try:
                return f"{int(val):,}".replace(",", ".")
            except Exception:
                return str(val)
        if isinstance(val, float):
            return f"{val:,.0f}"
        return str(val)

    table = html.Table(
        [
            html.Tr([html.Th("Sector"), html.Td(td_safe(info.get("sector")))]),
            html.Tr([html.Th("Industry"), html.Td(td_safe(info.get("industry")))]),
            html.Tr([html.Th("Employees"), html.Td(td_safe(info.get("employees"), binlik=True))]),
            html.Tr([html.Th("Earnings Date"), html.Td(td_safe(info.get("earnings_date")))]),
            html.Tr([html.Th("Market Cap"), html.Td(td_safe(info.get("market_cap"), binlik=True))]),
            html.Tr([html.Th("Radar"), html.Td(td_safe(info.get("radar")))]),
        ],
        className="table table-sm",
    )

    summary = html.Div(
        [html.H5("Summary"), html.Pre(info.get("summary") or "", style={"whiteSpace": "pre-wrap"})],
        className="p-2 border rounded",
    )
    # Grafikler
    charts = []
    fiyat_df = metrics_df[metrics_df.metric.isin(["Fiyat", "Tahmin"])]
    if not fiyat_df.empty:
        fig = go.Figure()
        for metric, color in zip(["Fiyat", "Tahmin"], ["#1976d2", "#a6761d"]):
            d = fiyat_df[fiyat_df.metric == metric]
            if not d.empty:
                fig.add_trace(go.Scatter(
                    x=d.period,
                    y=d.value,
                    mode="lines+markers",
                    name=metric,
                    line=dict(width=2, color=color),
                ))
        fig.update_layout(
            title="Fiyat & Tahmin",
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=1.12),
        )
        charts.append(
            dcc.Graph(
                figure=fig,
                className="chart-graph",
                config={"displayModeBar": False, "staticPlot": True}
            )
        )

    # Diğer metrikler
    for metric in [
        "MCap/CATS", "Capex/Amort", "EBIT Margin", "FCF Margin",
        "Gross Margin", "CATS", "Satış Çeyrek", "EBIT Çeyrek", "Net Kar Çeyrek"
    ]:
        df = metrics_df[metrics_df.metric == metric].dropna(subset=["period", "value"])
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.period,
                y=df.value,
                mode="lines+markers",
                name=metric,
                line=dict(width=2),
            ))
            fig.update_layout(
                title=metric,
                height=240,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            charts.append(
                dcc.Graph(
                    figure=fig,
                    className="chart-graph",
                    config={"displayModeBar": False, "staticPlot": True}
                )
            )

    charts_container = html.Div(charts)
    return html.Div([
        header,
        html.Hr(),
        table,
        summary,
        html.Hr(),
        charts_container
    ])

# ────────────── FAVORİ CALLBACK ──────────────
@app.callback(
    Output("store-favs", "data"),
    Input("fav-toggle", "n_clicks"),
    State("url", "href"),
    State("store-favs", "data"),
    prevent_initial_call=True,
)
def toggle_fav(n_clicks, href, favs):
    # İlk render / layout yenilemesinde n_clicks 0 ya da None olur ⇒ işlem yapma
    if not callback_context.triggered or not n_clicks:
        raise PreventUpdate

    ticker = parse_ticker_from_href(href)
    if not ticker:
        raise PreventUpdate

    favs = list(favs or [])      # kopya: Dash değişikliği algılasın
    if ticker in favs:
        favs.remove(ticker)      # yıldız doluyken → kaldır
    else:
        favs.append(ticker)      # yıldız boşken  → ekle

    return favs


def favorites_layout(favs):
    return html.Div(
        [
            html.H4("⭐ Favoriler"),
            html.Hr(),
            html.Ul([html.Li(make_company_link(t), className="mb-2") for t in favs]) if favs else html.Div("Favorilere eklenmiş şirket yok."),
        ]
    )

# ────────────── RADAR CALLBACK ──────────────
def radar_layout(radars):
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

@app.callback(
    Output("btn-refresh-data", "children"),
    Input("btn-refresh-data", "n_clicks"),
    prevent_initial_call=True,
)
def refresh_data(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    # RAM cache'i temizle!
    load_company_info.cache_clear()
    load_metrics.cache_clear()
    return "✅ Veriler güncellendi!"

# ────────────── MAIN ──────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
