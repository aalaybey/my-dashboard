# dash_company_dashboard.py
import os
import functools
import secrets
from urllib.parse import parse_qs, urlparse

import dash
import dash_auth
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
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
    q = sa_text("""
        SELECT ticker, filing_date, sub_industry
        FROM company_info
        WHERE radar = 1
        ORDER BY
            CASE WHEN filing_date IS NULL THEN 1 ELSE 0 END,
            filing_date DESC,
            ticker ASC
    """)
    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn)

    records = []
    for _, row in df.iterrows():
        fd = row["filing_date"]
        if pd.isnull(fd):
            fd_str = "-"
        else:
            try:
                fd_str = str(fd)[:10]
            except Exception:
                fd_str = str(fd)

        si = row.get("sub_industry")
        si_str = str(si).strip() if pd.notna(si) and str(si).strip() else "-"

        records.append({
            "ticker": row["ticker"],
            "filing_date": fd_str,
            "sub_industry": si_str
        })

    return records

@functools.lru_cache(maxsize=1)
def load_companies_grouped():
    """
    Tüm şirketleri industry'lere göre gruplar. Her industry içinde:
    - potential DESC (NULLS LAST)
    - eşitlikte ticker ASC
    Kolonlar: ticker, industry, potential, market_cap, price_last, pred_last
    """
    q = sa_text("""
        WITH latest AS (
            SELECT
                ticker,
                metric,
                value::numeric AS value,
                ROW_NUMBER() OVER (PARTITION BY ticker, metric ORDER BY period DESC) AS rn
            FROM excel_metrics
            WHERE metric IN ('Fiyat','Tahmin')
        )
        SELECT
            ci.ticker,
            COALESCE(ci.industry, 'Unknown') AS industry,
            ci.potential::numeric AS potential,
            ci.market_cap::numeric AS market_cap,
            /* Fiyat: SONDAN BİR ÖNCEKİ (rn=2) | Tahmin: EN YENİ (rn=1) */
            MAX(CASE WHEN l.metric='Fiyat'  AND l.rn=2 THEN l.value END) AS price_last,
            MAX(CASE WHEN l.metric='Tahmin' AND l.rn=1 THEN l.value END) AS pred_last
        FROM company_info ci
        LEFT JOIN latest l ON l.ticker = ci.ticker
        GROUP BY ci.ticker, industry, potential, market_cap
        ORDER BY industry, potential DESC NULLS LAST, ci.ticker ASC
    """)

    with get_engine().connect() as conn:
        df = pd.read_sql(q, conn)

    grouped = {}
    for ind, g in df.groupby("industry", sort=True):
        rows = g[["ticker", "potential", "market_cap", "price_last", "pred_last"]].to_dict(orient="records")
        grouped[ind] = rows
    return grouped



# ────────────── BİLEŞENLER ──────────────
def search_input():
    return dbc.Input(
        id="search-input",
        placeholder="Ticker ara…",
        type="text",
        debounce=True,
        style={"maxWidth": "220px"},
    )

def make_company_link(ticker):
    # Yeni sekmede aç
    return dcc.Link(ticker, href=f"/?t={ticker}", target="_blank")


def companies_layout():
    data = load_companies_grouped()

    def fmt_potential(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return str(x)

    def fmt_money_int(x):
        # Market cap gibi büyük sayılar için 1.234.567 formatı (virgül yerine nokta)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        try:
            return f"{int(float(x)):,}".replace(",", ".")
        except Exception:
            return str(x)

    def fmt_price(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        try:
            return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(x)

    blocks = [html.H4("🏢 Şirketler (Industry'lere göre, Potential ↓)"), html.Hr()]

    for industry, rows in data.items():
        # Tablo başlığı
        header = html.Thead(
            html.Tr([
                html.Th("Ticker"),
                html.Th("Potential"),
                html.Th("Market Cap"),
                html.Th("Fiyat (en yeni)"),
                html.Th("Tahmin (en yeni)"),
            ])
        )
        # Satırlar
        body_rows = []
        for r in rows:
            body_rows.append(
                html.Tr([
                    html.Td(make_company_link(r["ticker"])),
                    html.Td(fmt_potential(r.get("potential"))),
                    html.Td(fmt_money_int(r.get("market_cap"))),
                    html.Td(fmt_price(r.get("price_last"))),
                    html.Td(fmt_price(r.get("pred_last"))),
                ])
            )

        table = dbc.Table(
            [header, html.Tbody(body_rows if body_rows else [html.Tr([html.Td("Kayıt yok", colSpan=5)])])],
            bordered=False,
            hover=True,
            responsive=True,
            striped=True,
            className="table-sm mb-4"
        )

        blocks.append(html.H5(industry))
        blocks.append(table)

    return html.Div(blocks)



# ────────────── LAYOUT ──────────────
app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="store-radar", storage_type="session"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        search_input(),
                    ],
                    sm=6,
                    className="text-start",
                ),
                dbc.Col(
                    [
                        dbc.Button("Ana Sayfa", href="https://alaybey.onrender.com/", color="primary", outline=True,
                                   className="ms-1 mb-2", id="btn-home"),
                        dbc.Button("Radar", id="btn-radar", color="secondary", outline=True, className="ms-2 mb-2"),
                        html.A("Fit", href="/fit", target="_blank", className="btn btn-outline-info ms-2 mb-2"),
                        html.A("All", href="/all", target="_blank", className="btn btn-outline-info ms-2 mb-2"),
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
    [Input("url", "href"), Input("btn-radar", "n_clicks")],
    [State("store-radar", "data")],
)
def render_page(href, radar_click, radar_data):
    ctx = callback_context
    trig_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "url"
    radar_data = radar_data or []

    if trig_id == "btn-radar":
        return radar_layout(radar_data)

    ticker = parse_ticker_from_href(href)
    if not ticker:
        return html.Div()
    return company_layout(ticker)


# ────────────── ŞİRKET SAYFASI ──────────────
def company_layout(ticker):
    info = load_company_info(ticker)
    if info is None:
        return html.Div("Geçersiz ticker.")
    metrics_df = load_metrics(ticker)

    header = dbc.Row(
        [
            dbc.Col(html.H4(ticker, className="mb-0"), width="auto"),
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
            html.Tr([html.Th("Sub Industry"), html.Td(td_safe(info.get("sub_industry")))]),
            html.Tr([html.Th("Employees"), html.Td(td_safe(info.get("employees"), binlik=True))]),
            html.Tr([html.Th("Earnings Date"), html.Td(td_safe(info.get("earnings_date")))]),
            html.Tr([html.Th("Filing Date"), html.Td(td_safe(info.get("filing_date")))]),
            html.Tr([html.Th("Market Cap"), html.Td(td_safe(info.get("market_cap"), binlik=True))]),
            html.Tr([html.Th("Potential"),
                     html.Td(f"{info.get('potential'):.2f}" if info.get("potential") is not None else "-")]),

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
        # --- Hazırlık
        # NULL period / value olanları at ve sıralayıp index’i düzelt
        d_price = (fiyat_df[fiyat_df.metric == "Fiyat"]
                   .dropna(subset=["period", "value"])
                   .sort_values("period").reset_index(drop=True))
        d_pred = (fiyat_df[fiyat_df.metric == "Tahmin"]
                  .dropna(subset=["period", "value"])
                  .sort_values("period").reset_index(drop=True))

        def signed_log(arr):
            arr = np.array(pd.to_numeric(arr, errors="coerce"), dtype=float, copy=True)
            # 0 değerlerini çizmiyoruz (log tanımsız)
            arr[arr == 0] = np.nan
            return np.sign(arr) * np.log10(np.abs(arr))

        # 1) Fiyat: pozitif log (TradingView tarzı, base10)
        y_price = signed_log(d_price["value"].values)
        pos_max = np.nanmax(y_price) if np.isfinite(np.nanmax(y_price)) else 0.0

        # 2) Tahminleri signed-log’a çevir (negatif varsa alt tarafa düşer)
        y_pred_raw = signed_log(d_pred["value"].values)

        # Ayna/simetri SADECE negatif tahmin varsa devrede
        has_neg_pred = pd.to_numeric(d_pred["value"], errors="coerce").lt(0).any()

        if has_neg_pred:
            # Aynayı aç: negatif limit = pozitif limitin simetriği
            lower_bound = -pos_max
            upper_bound = pos_max
            # Sadece bu modda negatifte dışarı taşanları kliple
            y_pred = np.where(y_pred_raw < lower_bound, lower_bound, y_pred_raw)
        else:
            # Aynayı kapat: tek eksen log (fiyat + tahmin) ve KLİP YOK
            # Alt sınırı her iki serinin minimumuna göre belirle
            min_pos = np.nanmin([np.nanmin(y_price), np.nanmin(y_pred_raw)])
            lower_bound = int(np.floor(min_pos)) if np.isfinite(min_pos) else 0
            upper_bound = pos_max
            y_pred = y_pred_raw

        # 4) Pozitif tarafta fiyat üstünü aşan tahmin varsa, ekseni genişlet (güvenli)
        pred_max = None
        try:
            pred_max = float(np.nanmax(y_pred_raw))
        except (ValueError, TypeError):
            pred_max = None

        if pred_max is not None and pred_max > upper_bound:
            upper_bound = pred_max

        # Üst limiti güvenli yuvarla (taşmayı engelle)
        k_min = int(np.floor(lower_bound)) if np.isfinite(lower_bound) else -1
        k_max = int(np.ceil(upper_bound)) if np.isfinite(upper_bound) else 1

        # Tick değerleri (…,-100,-10,0,10,100,…) — normal sayı olarak yaz
        tickvals = list(range(k_min, k_max + 1))
        ticktext = []
        for k in tickvals:
            if k == 0:
                ticktext.append("0")
            else:
                ticktext.append(str(int(np.sign(k) * (10 ** abs(k)))))

        # Y-ekseni aralığı: integer log-güç aralığına sabitle
        ylim = (k_min, k_max)

        # --- Çizim
        fig = go.Figure()

        # Fiyat (mavi): pozitif log tarafı
        fig.add_trace(go.Scatter(
            x=d_price["period"], y=y_price,
            mode="lines", name="Fiyat",
            line=dict(width=2, color="#1976d2")
        ))
        # Son FİYAT noktasını etiketle (son GEÇERLİ noktaya; mavi ve üstte)
        valid_idx_price = np.where(np.isfinite(y_price))[0]
        if valid_idx_price.size:
            i = int(valid_idx_price[-1])
            last_price_text = f"{float(d_price['value'].iloc[i]):,.2f}"
            fig.add_trace(go.Scatter(
                x=[d_price['period'].iloc[i]], y=[y_price[i]],
                text=[last_price_text],
                mode="text",
                textposition="middle right",  # tahminle aynı yana
                textfont=dict(size=11, color="#1976d2"),  # mavi görünsün
                hoverinfo="skip",
                showlegend=False
            ))

        # Tahmin (turuncu): nokta + çizgi, klip uygulanmış değerlerle
        fig.add_trace(go.Scatter(
            x=d_pred["period"], y=y_pred,
            mode="lines+markers", name="Tahmin",
            line=dict(width=2, color="#a6761d"),
            marker=dict(size=5)
        ))
        # Son tahmin noktasını etiketle (orijinal değeri yaz)
        if len(d_pred) > 0 and np.isfinite(y_pred[-1]):
            last_text = f"{float(d_pred['value'].iloc[-1]):,.2f}"
            fig.add_trace(go.Scatter(
                x=[d_pred['period'].iloc[-1]], y=[y_pred[-1]],
                text=[last_text], mode="text",
                textposition="middle right",
                showlegend=False
            ))

        # Kategorileri sadece GEÇERLİ period’lardan oluştur (None/NaT yok)
        cats_price = d_price["period"].astype(str).tolist()
        cats_pred = d_pred["period"].astype(str).tolist()
        cats = pd.Index(cats_price).union(pd.Index(cats_pred))  # temiz birleşim

        # Sağdan taşmayı kesmek için küçük ped
        right_pad = 0.02
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=list(cats),
            range=[-0.5, len(cats) - right_pad],
            automargin=True
        )

        fig.update_layout(
            title="Fiyat & Tahmin (Signed-Log, Aynalı Eksen)",
            height=560,
            # alttan taşmayı kesin bitirmek için alt marjı büyüt
            margin=dict(l=12, r=32, t=80, b=160),
            # lejandı GRAFİĞİN İÇİNDE alt tarafa al (y=0.02)
            legend=dict(orientation="h", y=-0.2, x=0.01, xanchor="left")
        )

        fig.update_yaxes(
            range=[k_min, k_max],  # taşmayı önlemek için tamsayı güç aralığı
            tickvals=tickvals,
            ticktext=ticktext,
            zeroline=True,
            zerolinewidth=1
        )

        charts.append(
            dcc.Graph(
                figure=fig,
                className="chart-graph",
                style={"marginTop": "8px"},  # üstte minicik boşluk
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

# ────────────── RADAR CALLBACK ──────────────
def radar_layout(radars):
    items = []

    for item in (radars or []):
        if isinstance(item, dict):
            ticker = item.get("ticker")
            filing_date = item.get("filing_date", "-")
            sub_industry = item.get("sub_industry", "-")
        else:
            ticker = item
            filing_date = "-"
            sub_industry = "-"

        items.append(
            html.Li(
                [
                    make_company_link(ticker),
                    html.Span(f" — {filing_date}", style={"marginLeft": "8px", "color": "#666"}),
                    html.Span(f" — {sub_industry}", style={"marginLeft": "8px", "color": "#999"})
                ],
                className="mb-2"
            )
        )

    return html.Div(
        [
            html.H4("🕵️ Radar Listesi"),
            dbc.Button("Listeyi Güncelle", id="btn-update-radar", color="info", outline=True, size="sm", className="mb-2"),
            html.Hr(),
            html.Ul(items) if items else html.Div("Radar'da şirket yok."),
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
    load_company_info.cache_clear()
    load_metrics.cache_clear()
    load_companies_grouped.cache_clear()
    return "✅ Veriler güncellendi!"


# ────────────── RADAR HTML SAYFALARI ──────────────
from flask import Response as FlaskResponse

@app.server.route("/fit")
def serve_fit():
    try:
        with get_engine().connect() as conn:
            row = conn.execute(sa_text("SELECT html FROM radar_fit WHERE id = 1")).fetchone()
        if row and row[0]:
            return FlaskResponse(row[0], content_type="text/html; charset=utf-8")
    except Exception as e:
        return FlaskResponse(f"Hata: {e}", status=500)
    return FlaskResponse("Fit radar verisi henüz yüklenmemiş.", status=404)

@app.server.route("/all")
def serve_all():
    try:
        with get_engine().connect() as conn:
            row = conn.execute(sa_text("SELECT html FROM radar_all WHERE id = 1")).fetchone()
        if row and row[0]:
            return FlaskResponse(row[0], content_type="text/html; charset=utf-8")
    except Exception as e:
        return FlaskResponse(f"Hata: {e}", status=500)
    return FlaskResponse("All radar verisi henüz yüklenmemiş.", status=404)


# ────────────── MAIN ──────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
