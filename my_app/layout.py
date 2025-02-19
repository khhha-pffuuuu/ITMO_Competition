from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
from config import MAIN_BG_COLOR, BORDER_COLOR, CARD_BG_COLOR, TEXT_COLOR, ACCENT_COLOR, PLACEHOLDER_COLOR

def serve_layout():
    layout = dbc.Container([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-init', style={'display': 'none'}),
        # Хранилища для данных и истории чата
        dcc.Store(id='stored-file', data={'contents': None, 'filename': None}),
        dcc.Store(id='data-processed', data=False),
        dcc.Store(id='chat-history', data=[]),
        dcc.Store(id="processed-dataset", data=None),

        # Заголовок приложения
        html.H2(
            "Анализ тональности",
            style={
                "textAlign": "center",
                "color": TEXT_COLOR,
                "marginBottom": "20px",
                'fontWeight': 'bold'
            }
        ),

        # Вкладки: данные и чат
        dbc.Tabs(
            id="tabs-container",
            active_tab="data-tab",
            children=[
                dbc.Tab(
                    label="Разметка данных",
                    tab_id="data-tab",
                    children=[
                        # Панель инструментов: загрузка файла, очистка и запуск анализа (статистика)
                        dbc.Row(
                            [
                                # Поле загрузки файла
                                dbc.Col(
                                    dcc.Upload(
                                        id="upload-data",
                                        children=html.Div([
                                            html.A("Выберите файл (CSV/XLSX)", style={"color": TEXT_COLOR})
                                        ]),
                                        style={
                                            "width": "100%",
                                            "height": "50px",
                                            "lineHeight": "50px",
                                            "borderRadius": "50px",
                                            "textAlign": "center",
                                            "backgroundColor": CARD_BG_COLOR,
                                            "color": TEXT_COLOR,
                                            "cursor": "pointer"
                                        },
                                        style_active={
                                            "backgroundColor": "#505050",
                                            "transition": "0.1s background-color"
                                        },
                                        multiple=False
                                    ),
                                    width=7
                                ),

                                # Кнопка сброса
                                dbc.Col(
                                    dbc.Button(
                                        DashIconify(icon="mdi:trash-can-outline", color="white", width=20),
                                        id="reset-button",
                                        className="w-100",
                                        style={
                                            "backgroundColor": MAIN_BG_COLOR,
                                            "borderRadius": "50px",
                                            "color": "white",
                                            "height": "50px",
                                            "border": f"1px solid {BORDER_COLOR}"
                                        }
                                    ),
                                    width=2
                                ),
                                dbc.Tooltip("Очистить загруженные данные", target="reset-button", placement="top"),

                                # Кнопка анализа
                                dbc.Col(
                                    dbc.Button(
                                        DashIconify(icon="mdi:chart-bar", color="white", width=20),
                                        id="analyze-button",
                                        className="w-100 no-focus",
                                        n_clicks=0,
                                        style={
                                            "backgroundColor": MAIN_BG_COLOR,
                                            "borderRadius": "50px",
                                            "color": "white",
                                            "height": "50px",
                                            "border": f"1px solid {BORDER_COLOR}"
                                        }
                                    ),
                                    width=2
                                ),
                                dbc.Tooltip("Подсчитать статистику", target="analyze-button", placement="top"),

                                # Кнопка скачивания
                                dbc.Col(
                                    dbc.Button(
                                        DashIconify(icon="mdi:download", width=18),
                                        id="download-dataset-btn",
                                        className="no-focus",
                                        color="primary",
                                        disabled=True,
                                        style={
                                            "textAlign": "center",
                                            "backgroundColor": TEXT_COLOR,
                                            "color": MAIN_BG_COLOR,
                                            "borderRadius": "50px",
                                            "height": "50px",
                                            'padding': '0 15px'
                                        }
                                    ),
                                    width=1
                                ),
                                dbc.Tooltip("Скачайте обработанный файл", target="download-dataset-btn", placement="top"),

                                # Компонент скачивания
                                dcc.Download(id="download-dataset"),
                            ],
                            className="mt-3 mb-3"
                        ),

                        # Область отображения данных
                        html.Div(
                            id='output-data-container',
                            children=[
                                html.Div(
                                    "Нет данных...",
                                    id="no-data-text",
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                        "height": "100%",
                                        "color": PLACEHOLDER_COLOR,
                                        "fontSize": "18px"
                                    }
                                ),
                                html.Div(
                                    id='output-data-upload',
                                    style={
                                        "display": "none",
                                        "overflowY": "auto",
                                        "height": "100%",
                                        "padding": "10px",
                                        "boxSizing": "border-box"
                                    }
                                )
                            ],
                            style={
                                "backgroundColor": CARD_BG_COLOR,
                                "borderRadius": "10px",
                                "border": f"1px solid {BORDER_COLOR}",
                                "height": "400px",
                                "overflow": "hidden",
                                "position": "relative",
                                "maxWidth": "100%"
                            }
                        ),
                        # html.Div(
                        #     children=[
                        #         # Кнопка скачать и Download
                        #         dbc.Button(DashIconify(icon="mdi:download", width=18),
                        #                    id="download-dataset-btn",
                        #                    color="primary",
                        #                    style={
                        #                        'textAlign': 'center',
                        #                        'backgroundColor': TEXT_COLOR,
                        #                        'color': MAIN_BG_COLOR,
                        #                        'borderRadius': '30px',
                        #                        'padding': '10px 20px'
                        #                    }),
                        #         dcc.Download(id="download-dataset")
                        #     ],
                        #     style={
                        #         "width": '100%',
                        #         'display': 'flex',
                        #         'justifyContent': 'right',
                        #         'alignItems': 'center',
                        #         "marginTop": "10px"
                        #     }
                        # ),
                    ]
                ),
                dbc.Tab(
                    label="Чат",
                    tab_id="chat-tab",
                    children=[
                        html.Div(
                            id="chat-messages",
                            children=[],
                            style={
                                "backgroundColor": CARD_BG_COLOR,
                                "borderRadius": "10px",
                                "padding": "10px 20px",
                                "border": f"1px solid {BORDER_COLOR}",
                                "maxWidth": "100%",
                                "height": "400px",
                                "overflowY": "auto",
                                "color": TEXT_COLOR,
                                "marginTop": "16px",
                                "scrollbarWidth": "thin",
                                "scrollbarColor": f"{BORDER_COLOR} {CARD_BG_COLOR}",
                            }
                        ),
                        dbc.InputGroup([
                            dbc.Input(
                                id="chat-input",
                                placeholder="Введите сообщение...",
                                n_submit=0,
                                className="no-focus",
                                style={
                                    "borderRadius": "30px 0 0 30px",
                                    "backgroundColor": CARD_BG_COLOR,
                                    "color": TEXT_COLOR,
                                    "border": f"1px solid {BORDER_COLOR}",
                                    'height': '50px',
                                    'lineHeight': '50px',
                                    "paddingLeft": '20px'
                                }
                            ),
                            dbc.Button(
                                DashIconify(icon="mdi:send", color=TEXT_COLOR, width=24),  # "Отправить",
                                id="chat-send",
                                color="secondary",
                                style={
                                    "paddingLeft": "20px",
                                    "paddingRight": "20px",
                                    "backgroundColor": CARD_BG_COLOR,
                                    "borderRadius": "0 30px 30px 0",  # округление только справа
                                    "border": f"1px solid {BORDER_COLOR}"
                                }
                            )
                        ], style={"marginTop": "20px"})
                    ]
                )
            ],
            style={"width": "100%", "marginTop": "10px"}
        ),

        # Модальное окно со статистикой
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle(" Статистика по сообщениям", style={"color": TEXT_COLOR})
                ),
                dbc.ModalBody(
                    [
                        dcc.Graph(id='pie-chart'),
                        html.Hr(),
                        html.Div(id='other-metrics')
                    ],
                    style={"backgroundColor": CARD_BG_COLOR, "color": TEXT_COLOR}
                ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Закрыть",
                        id="close-modal",
                        className="ms-auto",
                        style={"backgroundColor": ACCENT_COLOR, "borderRadius": "10px", "border": "none"}
                    )
                )
            ],
            id="modal",
            is_open=False,
            size="lg"
        )
    ], fluid=True, style={"backgroundColor": MAIN_BG_COLOR, "width": "80%", "marginTop": "20px"})

    return layout
