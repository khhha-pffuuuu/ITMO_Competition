import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from config import MAIN_BG_COLOR, BORDER_COLOR, CARD_BG_COLOR, TEXT_COLOR, ACCENT_COLOR, PLACEHOLDER_COLOR
from utils import parse_contents, classes_counts


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
