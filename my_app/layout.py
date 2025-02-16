from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
from config import MAIN_BG_COLOR, BORDER_COLOR, CARD_BG_COLOR, TEXT_COLOR, ACCENT_COLOR, PLACEHOLDER_COLOR


def serve_layout():
    layout = dbc.Container([
        # Для отслеживания перехода по URL
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-init', style={'display': 'none'}),
        # Хранилища для данных
        dcc.Store(id='stored-file', data={'contents': None, 'filename': None}),
        dcc.Store(id='data-processed', data=False),

        # Заголовок приложения
        html.H2("Анализ тональности",
                style={"textAlign": "center", "color": TEXT_COLOR, "marginBottom": "20px", 'fontWeight': 'bold'}),

        # Панель инструментов: загрузка, сброс, анализ
        dbc.Row([
            dbc.Col(
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        '📂 ',
                        html.A('Выберите файл (CSV/XLSX)', style={"color": TEXT_COLOR})
                    ]),
                    style={
                        'width': '100%',
                        'height': '50px',
                        'lineHeight': '50px',
                        'borderRadius': '50px',
                        'textAlign': 'center',
                        'backgroundColor': CARD_BG_COLOR,
                        'color': TEXT_COLOR,
                    },
                    style_active={
                        'backgroundColor': '#505050',
                        'transition': '0.1 background-color'
                    },
                    multiple=False
                ),
                width=8
            ),
            dbc.Col(
                dbc.Button(
                    [DashIconify(icon="mdi:trash-can-outline", color="white", width=20)],
                    id="reset-button",
                    className="w-100",
                    style={
                        "backgroundColor": MAIN_BG_COLOR,
                        "borderRadius": "50px",
                        "color": "white",
                        "height": "50px",
                        'border': f"1px solid {BORDER_COLOR}"
                    }
                ),
                width=2
            ),
            dbc.Col(
                dbc.Button(
                    [DashIconify(icon="mdi:chart-bar", color="white", width=20)],
                    id="analyze-button",
                    className="w-100",
                    n_clicks=0,
                    style={
                        "backgroundColor": MAIN_BG_COLOR,
                        "borderRadius": "50px",
                        "color": "white",
                        "height": "50px",
                        'border': f"1px solid {BORDER_COLOR}"
                    }
                ),
                width=2
            ),
        ], className="mt-3 mb-3",
            style={"borderBottom": f"1px solid {BORDER_COLOR}", "paddingBottom": "15px"}),

        # Область отображения данных
        html.Div(id='output-data-container', children=[
            html.Div("Нет данных...", id="no-data-text", style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "height": "100%",
                "color": PLACEHOLDER_COLOR,
                "fontSize": "18px"
            }),
            html.Div(
                id='output-data-upload',
                style={
                    "display": "none",
                    "height": "100%",
                    "overflow": "auto",
                    "position": "relative",
                    "scrollbarWidth": "thin",
                    "scrollbarColor": f"{BORDER_COLOR} {CARD_BG_COLOR}",
                }
            )
        ], style={
            "backgroundColor": CARD_BG_COLOR,
            "borderRadius": "10px",
            "padding": "10px",
            "border": f"1px solid {BORDER_COLOR}",
            "height": "65vh",
            "maxWidth": "100%",
            "overflow": "hidden",
        }),

        # Модальное окно со статистикой
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("📊 Статистика по сообщениям", style={"color": TEXT_COLOR})),
            dbc.ModalBody([
                dcc.Graph(id='pie-chart'),
                html.Hr(),
                html.Div(id='other-metrics')
            ], style={"backgroundColor": CARD_BG_COLOR, "color": TEXT_COLOR}),
            dbc.ModalFooter(
                dbc.Button("Закрыть", id="close-modal", className="ms-auto",
                           style={"backgroundColor": ACCENT_COLOR, "borderRadius": "10px", "border": "none"})
            )
        ], id="modal", is_open=False, size="lg")
    ], fluid=True, style={"backgroundColor": MAIN_BG_COLOR, "width": "80%", "marginTop": "20px"})

    return layout
