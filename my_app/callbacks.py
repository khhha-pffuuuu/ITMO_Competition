import pandas as pd

import dash
from dash import html, dcc, callback_context, ALL
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from config import MAIN_BG_COLOR, BORDER_COLOR, CARD_BG_COLOR, TEXT_COLOR, ACCENT_COLOR, PLACEHOLDER_COLOR
from utils import parse_contents, classes_counts, predict


def register_callbacks(app, model, tokenizer):
    # При переходе по URL сбрасываем глобальную переменную
    @app.callback(
        Output('page-init', 'children'),
        Input('url', 'pathname')
    )
    def initialize_page(pathname):
        global classes_counts
        classes_counts = None
        return ""

    @app.callback(
        [Output('stored-file', 'data'),
         Output('upload-data', 'contents')],
        [Input('upload-data', 'contents'),
         Input('reset-button', 'n_clicks')],
        [State('upload-data', 'filename'),
         State('stored-file', 'data')]
    )
    def update_store(contents, reset_clicks, filename, stored_data):
        ctx = callback_context
        global classes_counts
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'reset-button':
            classes_counts = None
            return {'contents': None, 'filename': None}, None
        elif trigger_id == 'upload-data' and contents is not None:
            return {'contents': contents, 'filename': filename}, contents
        return stored_data, contents

    @app.callback(
        [Output('output-data-upload', 'children'),
         Output('output-data-upload', 'style'),
         Output('no-data-text', 'style'),
         Output('data-processed', 'data')],
        Input('stored-file', 'data')
    )
    def update_output(stored_data):
        global classes_counts
        if stored_data and stored_data.get('contents'):
            df, stats = parse_contents(stored_data['contents'], stored_data['filename'], model, tokenizer)
            classes_counts = stats

            if df is not None:
                table = dbc.Table.from_dataframe(
                    df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    style={
                        "color": TEXT_COLOR,
                        "backgroundColor": CARD_BG_COLOR,
                        "width": "100%",
                        "tableLayout": "auto",
                    }
                )
                return (
                    html.Div([
                        html.Div(table, style={
                            "overflow": "auto",
                            "height": "calc(65vh - 40px)",
                            "scrollbarWidth": "thin",
                            "scrollbarColor": f"{BORDER_COLOR} {CARD_BG_COLOR}",
                        })
                    ]),
                    {"display": "block"},
                    {"display": "none"},
                    True
                )
        return html.Div(), {"display": "none"}, {
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "height": "100%",
            "color": PLACEHOLDER_COLOR,
            "fontSize": "18px"
        }, False

    @app.callback(
        Output("modal", "is_open"),
        [Input("analyze-button", "n_clicks"),
         Input("close-modal", "n_clicks")],
        State("modal", "is_open")
    )
    def toggle_modal(n_open_analyze, n_close, is_open):
        ctx = callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == "analyze-button":
            return not is_open
        elif trigger_id == "close-modal":
            return False
        return is_open

    @app.callback(
        Output("pie-chart", "figure"),
        Input("modal", "is_open")
    )
    def update_pie_chart(is_open):
        global classes_counts
        if is_open and classes_counts is not None:
            items = sorted(classes_counts.items(), key=lambda x: x[1], reverse=True)
            if len(items) > 3:
                items = items[:3]
            labels, values = zip(*items)
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
            fig.update_layout(
                title="Распределение классов",
                paper_bgcolor=CARD_BG_COLOR,
                plot_bgcolor=CARD_BG_COLOR,
                font_color=TEXT_COLOR
            )
            return fig
        return go.Figure()

    @app.callback(
        [Output("analyze-button", "disabled"),
         Output("analyze-button", "style")],
        [Input("stored-file", "data"),
         Input("data-processed", "data")]
    )
    def update_analyze_button(stored_data, data_processed):
        base_style = {
            "backgroundColor": MAIN_BG_COLOR,
            "borderRadius": "50px",
            "color": "white",
            "height": "50px",
            'border': f"1px solid {BORDER_COLOR}"
        }
        if stored_data is None or stored_data.get("contents") is None or not data_processed:
            disabled_style = base_style.copy()
            disabled_style["backgroundColor"] = "#1a1a1a"
            return True, disabled_style
        return False, base_style

    # Callback для отправки сообщений в чат
    @app.callback(
        [Output("chat-history", "data"),
         Output("chat-messages", "children"),
         Output("chat-input", "value")],
        [Input("chat-send", "n_clicks"),
         Input("chat-input", "n_submit")],
        [State("chat-input", "value"),
         State("chat-history", "data")],
        prevent_initial_call=True
    )
    def update_chat(n_clicks, n_submit, new_message, history):
        # Проверяем, что что-то триггернулось
        if not callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        # Если ничего не введено или введена строка из пробелов, не обновляем
        if new_message is None or new_message.strip() == "":
            raise dash.exceptions.PreventUpdate

        # Предсказываем тональность сообщения
        msg_to_df = pd.DataFrame({'MessageText': [new_message.strip()]})
        prediction = predict(model, tokenizer, msg_to_df)

        # Если истории ещё нет, инициализируем список
        history = history or []
        history.append((new_message.strip(), int(prediction[0])))

        sentiment_colors = {
            0: "#FF4C4C",  # Красный (негатив)
            1: "#A0A0A0",  # Серый (нейтральный)
            2: "#4CAF50"  # Зеленый (позитив)
        }

        # Формируем список для отображения сообщений
        messages = []
        for msg, pred in history:
            border_color = sentiment_colors[pred]
            messages.append(
                html.Div([
                    html.Div([  # Контейнер для сообщения и смайликов
                        html.Div(msg, style={  # Само сообщение
                            "backgroundColor": ACCENT_COLOR,
                            "padding": "15px 30px",
                            "borderRadius": "30px",
                            "fontSize": "16px",
                            "color": TEXT_COLOR,
                            "flex": "1",
                            "wordWrap": "break-word",
                            "whiteSpace": "normal",
                            "minWidth": "0",
                            "maxWidth": "calc(100% - 150px)",
                            "overflow-wrap": "break-word"
                        }),
                        html.Div([  # Контейнер для трех смайликов
                            html.Div(id="emoji-container", children=[
                                html.Span("😡", id={"type": "emoji", "index": f"{msg}-angry"}, n_clicks=0, style={
                                    "padding": "10px",
                                    "cursor": "pointer",
                                    "color": "#f44336",
                                    "fontSize": "20px"
                                }),
                                html.Span("😐", id={"type": "emoji", "index": f"{msg}-neutral"}, n_clicks=0, style={
                                    "padding": "10px",
                                    "cursor": "pointer",
                                    "color": "#ffeb3b",
                                    "fontSize": "20px"
                                }),
                                html.Span("😊", id={"type": "emoji", "index": f"{msg}-happy"}, n_clicks=0, style={
                                    "padding": "10px",
                                    "cursor": "pointer",
                                    "color": "#4caf50",
                                    "fontSize": "20px"
                                })
                            ], style={  # Стиль контейнера для смайликов
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "space-between",
                                "backgroundColor": "#e0e0e0",
                                "borderRadius": "15px",
                                "padding": "5px",
                                "width": "100%",
                                "marginLeft": "10px"
                            })
                        ])
                    ], style={  # Общий стиль для сообщения и кнопок
                        "display": "flex",
                        "alignItems": "flex-start",
                        "flex-wrap": "wrap",
                        "backgroundColor": ACCENT_COLOR,
                        "padding": "15px 30px",
                        "borderRadius": "30px",
                        "fontSize": "16px",
                        "color": TEXT_COLOR,
                        "flex": "1"
                    }),
                    html.Div(["😡" if pred == 0 else "😐" if pred == 1 else "😊"], style={  # Текущая эмоция (неизменяемая)
                        "minWidth": "50px",
                        "textAlign": "center",
                        "fontSize": "20px",
                        "color": border_color,
                        "margin-left": "10px"
                    })
                ], style={  # Контейнер для сообщения и предсказания
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "10px",
                    "borderLeft": f"5px solid {border_color}",
                    "padding": "10px",
                    "marginBottom": "10px"
                })
            )
            
        return history, messages, ""  # Очищаем поле ввода после отправки

    