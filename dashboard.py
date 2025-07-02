import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback_context
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import os
from dash.exceptions import PreventUpdate
import dash_auth

# ---------------- SECRETS / LOGIN ----------------
VALID_USERNAME_PASSWORD_PAIRS = {
    os.environ.get("DASH_USER", "user"): os.environ.get("DASH_PASS", "1234")
}
# dash_auth ile basic login
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)

server = app.server  # deploy için

# ----------------- DATABASE CONNECT ----------------
def get_db_conn():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASS"),
        port=os.environ.get("DB_PORT"),
    )

def load_company_info():
    conn = get_db_conn()
    df = pd.read_sql("SELECT * FROM company_info", conn)
    conn.close()
    return df

def load_metrics():
    conn = get_db_conn()
    df = pd.read_sql("SELECT * FROM excel_metrics", conn)
    conn.close()
    return df

# ------------ Uygulama içi cache ve state için global değişkenler ----------
company_info = load_company_info()
metrics = load_metrics()
ALL_TICKERS = company_info['ticker'].sort_values().tolist()
METRICS_FOR_CHART = [
    "Fiyat", "Tahmin", "MCap/CATS", "Capex/Amort", "EBIT Margin",
    "FCF Margin", "Gross Margin", "CATS", "Satış Çeyrek", "EBIT Çeyrek", "Net Kar Çeyrek"
]

# --------------- Layout -----------------
def company_dropdown(selected=None):
    return dcc.Dropdown(
        options=[{"label": t, "value": t} for t in ALL_TICKERS],
        value=selected or (ALL_TICKERS[0] if ALL_TICKERS else None),
        id="company-select",
        style={"width": "100%", "margin-bottom": "10px"}
    )

def star_icon(fav):
    return html.I(className="bi bi-star-fill" if fav else "bi bi-star",
                  style={"font-size": "2rem", "color": "#f7c948", "cursor": "pointer"},
                  id="star-fav", n_clicks=0)

def get_summary_html(summary):
    # Summary'i satır kayması ile göster
    if not summary: return ""
    return html.Div(summary, style={"font-size": "1rem", "color": "#444", "whiteSpace": "pre-line"})

# --------- Favori/Radar session için dash dcc.Store kullanılacak ----------
app.layout = dbc.Container([
    dcc.Store(id="fav-store", storage_type="session"),  # favoriler
    dcc.Store(id="radar-store", storage_type="session"), # radar listesi
    dcc.Location(id="url"),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("Şirket Dashboard", className="text-primary", style={"margin":"15px 0 5px"}),
                dbc.Input(id="search-bar", placeholder="Şirket ticker ara...", type="text", debounce=True),
            ]),
        ], width=8),
        dbc.Col([
            dbc.Button("Radar", id="go-radar", color="secondary", outline=True, style={"margin-right":"6px"}),
            dbc.Button("Favoriler", id="go-favs", color="secondary", outline=True)
        ], width=4, style={"text-align":"right"})
    ], align="center", style={"margin-bottom":"12px"}),
    html.Hr(),
    html.Div(id="main-content")
], fluid=True, style={"maxWidth": "700px"})

# ----------- CALLBACKS -----------

# Search + Dropdown filtreleme
@app.callback(
    Output("company-select", "options"),
    Output("company-select", "value"),
    Input("search-bar", "value"),
    State("company-select", "value"),
)
def filter_dropdown(search, current):
    filtered = [t for t in ALL_TICKERS if (not search or search.lower() in t.lower())]
    sel = filtered[0] if filtered and (current not in filtered) else current
    return [{"label": t, "value": t} for t in filtered], sel or (filtered[0] if filtered else None)

# Ana sayfa yönlendirme
@app.callback(
    Output("main-content", "children"),
    Input("url", "pathname"),
    Input("go-favs", "n_clicks"),
    Input("go-radar", "n_clicks"),
    Input("company-select", "value"),
    State("fav-store", "data"),
    State("radar-store", "data"),
    prevent_initial_call=True
)
def render_page(path, favs_click, radar_click, ticker, favs, radar):
    ctx = callback_context
    if not ctx.triggered: raise PreventUpdate

    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    # Ana company sayfası
    if trig in ["company-select", "url"] and ticker:
        return company_page_layout(ticker, favs or [])
    elif trig == "go-favs":
        return favorites_layout(favs or [])
    elif trig == "go-radar":
        return radar_layout(radar or [])
    else:
        return company_page_layout(ticker, favs or [])

# Şirket sayfası ve yıldız favori callback
def company_page_layout(ticker, favs):
    info = company_info[company_info['ticker'] == ticker].iloc[0]
    df = metrics[metrics['ticker'] == ticker]
    fav = ticker in favs

    return html.Div([
        html.Div([
            company_dropdown(selected=ticker),
            html.Span([
                star_icon(fav),
                html.Span(" Favorilere ekle/çıkar", style={"font-size":"1.2rem"})
            ], style={"float":"right", "margin-top":"-55px"})
        ]),
        dbc.Row([
            dbc.Col([
                html.Table([
                    html.Tr([html.Td(html.B("Sector")), html.Td(info['sector'])]),
                    html.Tr([html.Td(html.B("Industry")), html.Td(info['industry'])]),
                    html.Tr([html.Td(html.B("Employees")), html.Td(f"{info['employees']:,}")]),
                    html.Tr([html.Td(html.B("Earnings Date")), html.Td(str(info['earnings_date']))]),
                    html.Tr([html.Td(html.B("Market Cap")), html.Td(str(info['market_cap']))]),
                    html.Tr([html.Td(html.B("Radar")), html.Td(str(info['radar']))]),
                ], style={"font-size":"1.07rem"}),
            ], width=5),
            dbc.Col([
                html.B("Summary"),
                get_summary_html(info.get("summary", "")),
            ], width=7)
        ], style={"margin-bottom":"15px"}),
        html.Hr(),
        html.Div([
            *[plot_metric(ticker, m, df) for m in METRICS_FOR_CHART]
        ], style={"maxHeight":"70vh", "overflowY":"auto"})
    ])

def plot_metric(ticker, metric, df):
    d = df[df['metric']==metric].dropna(subset=['period','value'])
    d = d[d['value'].apply(lambda x: str(x).lower() != "none")]
    if d.empty: return html.Div()
    d = d.sort_values("period")
    # Fiyat & Tahmin
    if metric == "Fiyat":
        fiyat = d.copy()
        tahmin = df[df['metric']=="Tahmin"].dropna(subset=['period','value'])
        tahmin = tahmin[tahmin['value'].apply(lambda x: str(x).lower() != "none")].sort_values("period")
        if fiyat.empty and tahmin.empty: return html.Div()
        fig = go.Figure()
        if not fiyat.empty:
            fig.add_trace(go.Scatter(
                x=fiyat["period"], y=fiyat["value"].astype(float),
                mode='lines+markers', name="Fiyat", line=dict(color='royalblue')
            ))
        if not tahmin.empty:
            fig.add_trace(go.Scatter(
                x=tahmin["period"], y=tahmin["value"].astype(float),
                mode='lines+markers', name="Tahmin", line=dict(color='chocolate')
            ))
        fig.update_layout(title="Fiyat & Tahmin", margin=dict(l=20, r=20, t=30, b=20), height=270)
        return dcc.Graph(figure=fig, config={"displayModeBar": False})
    # Diğer metrikler
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["period"], y=d["value"].astype(float),
        mode='lines+markers', name=metric
    ))
    fig.update_layout(title=metric, margin=dict(l=20, r=20, t=30, b=20), height=220)
    return dcc.Graph(figure=fig, config={"displayModeBar": False})

# Favori ekle/çıkar
@app.callback(
    Output("fav-store", "data"),
    Input("star-fav", "n_clicks"),
    State("company-select", "value"),
    State("fav-store", "data"),
    prevent_initial_call=True
)
def toggle_fav(n, ticker, favs):
    if not ticker: raise PreventUpdate
    favs = favs or []
    if ticker in favs:
        favs.remove(ticker)
    else:
        favs.append(ticker)
    return favs

# Favoriler sayfası
def favorites_layout(favs):
    return html.Div([
        html.H4("⭐ Favoriler"),
        html.Hr(),
        html.Ul([
            html.Li(dcc.Link(t, href=f"/?ticker={t}", refresh=True), style={"font-size":"1.2rem"}) for t in favs
        ]) if favs else html.Div("Favorilere eklediğiniz şirket yok.")
    ])

# Radar algoritması: radar=1 olanlar
def radar_list():
    # Son gelen veriyle radar=1 olanları döndür
    df = company_info
    radar_ones = df[df['radar'] == 1]['ticker'].tolist()
    return radar_ones

# Radar güncelleme butonu ve sayfa
@app.callback(
    Output("radar-store", "data"),
    Input("go-radar", "n_clicks"),
    prevent_initial_call=True
)
def update_radar(n):
    return radar_list()

def radar_layout(radars):
    return html.Div([
        html.H4("🕵️ Radar Listesi"),
        dbc.Button("Güncelle", id="update-radar-btn", color="info", outline=True, size="sm", style={"margin-bottom":"8px"}),
        html.Hr(),
        html.Ul([
            html.Li(dcc.Link(t, href=f"/?ticker={t}", refresh=True), style={"font-size":"1.2rem"}) for t in radars
        ]) if radars else html.Div("Radar'da şirket yok.")
    ])

@app.callback(
    Output("radar-store", "data"),
    Input("update-radar-btn", "n_clicks"),
    prevent_initial_call=True
)
def manual_update_radar(n):
    return radar_list()

if __name__ == "__main__":
    app.run_server(debug=True)