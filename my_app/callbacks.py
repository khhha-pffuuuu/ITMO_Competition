from io import StringIO, BytesIO

import pandas as pd

import json

import dash
from dash import html, dcc, callback_context, ALL
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from flask import request

from config import MAIN_BG_COLOR, BORDER_COLOR, CARD_BG_COLOR, TEXT_COLOR, ACCENT_COLOR, PLACEHOLDER_COLOR, ID2CLS
from utils import get_user_model, parse_contents, predict, compute_stats, render_messages, start_fine_tuning

from db.db_utils import add_feedback, extract_and_delete_feedback


def register_callbacks(app):
    @app.callback(
        Output("user-initialized", "data"),
        Input("url", "pathname")
    )
    def check_user(_):
        try:
            response = requests.get("http://127.0.0.1:8050/init_user", timeout=5)
            if response.status_code in [200, 204]:
                return {"initialized": True}
        except requests.exceptions.RequestException:
            pass
        return {"initialized": False}

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
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'reset-button':
            return {'contents': None, 'filename': None}, None
        elif trigger_id == 'upload-data' and contents is not None:
            return {'contents': contents, 'filename': filename}, contents
        return stored_data, contents

    @app.callback(
        [
            Output('output-data-upload', 'children'),
            Output('output-data-upload', 'style'),
            Output('no-data-text', 'style'),
            Output('data-processed', 'data'),
            Output("processed-dataset", "data"),
        ],
        Input('stored-file', 'data')
    )
    def update_output(stored_data):
        if stored_data and stored_data.get('contents'):
            user_id = request.cookies.get("user_id", "default")
            tokenizer, model = get_user_model(user_id)

            df = parse_contents(stored_data['contents'], stored_data['filename'])

            # Предсказываем классы
            df['Class'] = predict(model, tokenizer, df)
            df.Class = df.Class.apply(lambda x: ID2CLS[x])

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
                        'margin': '0'
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
                    True,
                    df.to_json(orient="records")
                )
        # Если данных нет
        return html.Div(), {"display": "none"}, {
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "height": "100%",
            "color": PLACEHOLDER_COLOR,
            "fontSize": "18px"
        }, False, None

    @app.callback(
        Output("download-dataset", "data"),
        Input("download-dataset-btn", "n_clicks"),
        State("processed-dataset", "data"),
        prevent_initial_call=True
    )
    def download_dataset(n_clicks, df_json):
        if not df_json:
            raise dash.exceptions.PreventUpdate

        df = pd.read_json(StringIO(df_json))

        csv_data = df.to_csv(index=False, encoding="utf-8-sig")
        return dict(content=csv_data, filename=f"processed_dataset.csv")

    # Callback для управления доступностью кнопки скачивания
    @app.callback(
        Output("download-dataset-btn", "disabled"),
        Input("upload-data", "contents"),
        prevent_initial_call=True
    )
    def enable_download_button(contents):
        return contents is None  # Если данные загружены, кнопка активна

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
        Input("modal", "is_open"),
        State("processed-dataset", "data")
    )
    def update_pie_chart(is_open, df_json):
        if is_open and df_json is not None:
            df = pd.read_json(StringIO(df_json))
            stats = compute_stats(df)

            items = sorted(stats.items(), key=lambda x: x[1], reverse=True)
            if len(items) > 3:
                items = items[:3]

            class_, values = zip(*items)
            fig = go.Figure(data=[go.Pie(labels=class_, values=values, hole=0.3)])
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
         Input("chat-input", "n_submit"),
         Input({"type": "emoji", "msg_index": dash.dependencies.ALL, "index": dash.dependencies.ALL}, "n_clicks")],
        [State("chat-input", "value"),
         State("chat-history", "data")],
        prevent_initial_call=True
    )
    def update_chat(n_clicks, n_submit, emoji_clicks, new_message, history):
        """
        1. Отправляет сообщение (предсказывает тональность и обновляет историю).
        2. Обрабатывает клик по смайлику (обновляет историю и fine-tuning).
        3. Ререндерит чат после любого изменения.
        """
        if not callback_context.triggered:
            raise dash.exceptions.PreventUpdate

        user_id = request.cookies.get("user_id", "default")
        tokenizer, model = get_user_model(user_id)

        # Проверяем, что триггер вызван отправкой сообщения
        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if trigger_id in ["chat-send", "chat-input"]:
            # Если ничего не введено или введена строка из пробелов, не обновляем
            if new_message is None or new_message.strip() == "":
                raise dash.exceptions.PreventUpdate

            # Предсказываем тональность сообщения
            msg_to_df = pd.DataFrame({'MessageText': [new_message.strip()]})
            prediction = predict(model, tokenizer, msg_to_df)

            # Если истории ещё нет, инициализируем список
            history = history or []
            history.append([new_message.strip(), int(prediction[0]), True])

            # Очищаем поле ввода после отправки
            return history, render_messages(history), ""

        elif "type" in trigger_id:
            # Если история пуста, то не обновляем
            if history is None:
                raise dash.exceptions.PreventUpdate

            # Проверяем, на какой смайл кликнули
            try:
                triggered_dict = json.loads(trigger_id)  # { "type": "emoji", "msg_index": i, "index": j }
                msg_i = triggered_dict["msg_index"]
                cls_i = triggered_dict["index"]
            except (json.JSONDecodeError, KeyError, TypeError):
                raise dash.exceptions.PreventUpdate

            if 0 <= msg_i < len(history):
                # Скрываем окно смайлов
                history[msg_i][-1] = False
                add_feedback(user_id, history[msg_i][0], cls_i)

                # Проверяем, накопилось ли для данного пользователя 100 сообщений
                df_feedback = extract_and_delete_feedback(user_id)
                if df_feedback is not None:
                    start_fine_tuning(model, tokenizer, user_id, df_feedback)

            return history, render_messages(history), dash.no_update

        raise dash.exceptions.PreventUpdate
