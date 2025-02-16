import base64
import io
import pandas as pd
import dash
from dash import html, dcc, Output, Input, State, callback_context
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import plotly.graph_objs as go  # для построения графика

from toolkits.predict import load_model, predict

external_stylesheets = [dbc.themes.DARKLY]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Анализ тональности"

# Загружаем модель
tokenizer, model = load_model()

# Цветовая схема
MAIN_BG_COLOR = "#222222"
BORDER_COLOR = "#505050"
CARD_BG_COLOR = "#303030"
TEXT_COLOR = "#E0E0E0"
ACCENT_COLOR = "#505050"
PLACEHOLDER_COLOR = "#888888"

# Глобальная переменная для статистики классов (либо None, либо словарь)
classes_counts = None

app.layout = dbc.Container([
    # Для отслеживания перехода по URL (обновление страницы)
    dcc.Location(id='url', refresh=False),
    # Скрытый элемент для инициализации страницы
    html.Div(id='page-init', style={'display': 'none'}),
    # Store для загруженных данных
    dcc.Store(id='stored-file', data={'contents': None, 'filename': None}),
    # Store для флага, что данные обработаны
    dcc.Store(id='data-processed', data=False),

    # Заголовок
    html.H2("Анализ тональности",
            style={"textAlign": "center", "color": TEXT_COLOR, "marginBottom": "20px", 'fontWeight': 'bold'}),

    # Панель инструментов (загрузка, сброс, анализ)
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
            ), width=8
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
                # Стиль и доступность кнопки будут обновляться через callback
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

    # Окно для отображения данных (фиксированная высота со скроллом)
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

# При переходе по URL сбрасываем глобальную переменную
@app.callback(
    Output('page-init', 'children'),
    Input('url', 'pathname')
)
def initialize_page(pathname):
    global classes_counts
    classes_counts = None
    return ""

def parse_contents(contents, filename):
    """ Декодируем файл, загружаем в pandas, и получаем статистику классов. """
    global classes_counts
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xlsx' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None

        marked_df, counts = predict(model, tokenizer, df)
        classes_counts = counts  # обновляем статистику
        return marked_df
    except Exception as e:
        print("Ошибка при чтении файла:", e)
        return None

@app.callback(
    [Output('stored-file', 'data'),
     Output('upload-data', 'contents')],
    [Input('upload-data', 'contents'), Input('reset-button', 'n_clicks')],
    [State('upload-data', 'filename'), State('stored-file', 'data')]
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

# Вывод данных в таблицу и обновление флага обработки
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('output-data-upload', 'style'),
     Output('no-data-text', 'style'),
     Output('data-processed', 'data')],
    Input('stored-file', 'data')
)
def update_output(stored_data):
    if stored_data and stored_data.get('contents'):
        df = parse_contents(stored_data['contents'], stored_data['filename'])
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
                True  # данные успешно обработаны
            )
    # Если данных нет или обработка не удалась, флаг False
    return html.Div(), {"display": "none"}, {
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "height": "100%",
        "color": PLACEHOLDER_COLOR,
        "fontSize": "18px"
    }, False

# Обработка открытия/закрытия модального окна
@app.callback(
    Output("modal", "is_open"),
    [Input("analyze-button", "n_clicks"), Input("close-modal", "n_clicks")],
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

# Обновление графика. Он строится только, если данные обработаны (classes_counts обновлён)
@app.callback(
    Output("pie-chart", "figure"),
    Input("modal", "is_open")
)
def update_pie_chart(is_open):
    global classes_counts
    if is_open and classes_counts is not None:
        # Ограничиваем количество классов до 3, если их больше
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

# Управление состоянием кнопки анализа:
# Кнопка будет неактивна, если нет загруженных данных или данные ещё не обработаны.
@app.callback(
    [Output("analyze-button", "disabled"),
     Output("analyze-button", "style")],
    [Input("stored-file", "data"), Input("data-processed", "data")]
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
        disabled_style["backgroundColor"] = "#1a1a1a"  # фон чуть темнее
        return True, disabled_style
    return False, base_style


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False, threaded=True, mode="external")
