import base64
import io
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

import torch as t
from torch import nn
import torch.nn.functional as f

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dash import html
from config import MAIN_BG_COLOR, BORDER_COLOR, CARD_BG_COLOR, TEXT_COLOR, ACCENT_COLOR, PLACEHOLDER_COLOR


# Загрузка модели
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path + '/base_model')
    model = AutoModelForSequenceClassification.from_pretrained(model_path + '/base_model')

    return tokenizer, model


# Очистка текстов
def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    text = text.replace("\xa0", " ")

    return text.strip()


# Предсказание ответов для DataFrame
def predict(model, tokenizer, df):
    pred = np.empty(df.shape[0])

    model.eval()
    for i in range(df.shape[0]):
        text = html_to_text(df.iloc[i]['MessageText'])
        inputs = tokenizer(text, truncation=True, max_length=256, return_tensors='pt')
        with t.no_grad():
            logits = model(**inputs).logits
        probs = f.softmax(logits, dim=-1)
        pred[i] = t.argmax(probs, dim=-1).item()

    return pred


# Глобальные переменные для статистики классов и значений меток классов
classes_counts = None
idx2classes = {
    0: 'B',  # Отрицательный класс
    1: 'N',  # Нейтральный класс
    2: 'G'  # Положительный класс
}


def parse_contents(contents, filename, model, tokenizer):
    """
    Декодирует файл, загружает его в pandas. DataFrame и выполняет предсказание.
    Обновляет глобальную переменную classes_counts.
    """
    global classes_counts, idx2classes

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xlsx' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None

        # Предсказываем классы
        df['labels'] = predict(model, tokenizer, df)
        df.labels = df.labels.apply(lambda x: idx2classes[x])

        # Считаем статистику
        unique_classes, counts = np.unique(df.labels, return_counts=True)
        stats = {i: 0 for i in ['B', 'N', 'G']}
        for i in range(len(unique_classes)):
            stats[unique_classes[i]] = counts[i]

        classes_counts = stats  # обновляем статистику

        return df, classes_counts
    except Exception as e:
        print("Ошибка при чтении файла:", e)
        return None


def render_messages(history):
    sentiment_colors = {
        0: "#FF4C4C",  # негатив
        1: "#A0A0A0",  # нейтраль
        2: "#4CAF50"  # позитив
    }

    messages = []
    for i, (msg, pred, display_buttons) in enumerate(history):
        border_color = sentiment_colors[pred]
        messages.append(
            html.Div([
                html.Div([
                    html.Div(msg, style={
                        "backgroundColor": ACCENT_COLOR,
                        "padding": "15px 30px",
                        "borderRadius": "30px",
                        "fontSize": "18px",
                        "color": TEXT_COLOR,
                        "flex": "1",
                        "wordWrap": "break-word",
                        "overflowWrap": "anywhere",
                        "whiteSpace": "normal",
                        "display": "inline-block",
                        "maxWidth": "calc(100% - 150px)"
                    }),
                    html.Div(
                        id='emoji-picker',
                        children=[
                            html.Span("😡", id={"type": "emoji", "msg_index": i, "index": 0}, n_clicks=0,
                                      style={"cursor": "pointer", "marginRight": "7.5px", "fontSize": "20px"}),
                            html.Span("😐", id={"type": "emoji", "msg_index": i, "index": 1}, n_clicks=0,
                                      style={"cursor": "pointer", "marginRight": "7.5px", "fontSize": "20px"}),
                            html.Span("😊", id={"type": "emoji", "msg_index": i, "index": 2}, n_clicks=0,
                                      style={"cursor": "pointer", "fontSize": "20px"})
                        ], style={
                            'display': 'flex' if display_buttons else 'none',
                            "position": 'absolute',
                            'right': '30px',
                            'top': '21px',
                            "border": "1px solid #ccc",
                            "backgroundColor": TEXT_COLOR,
                            "borderRadius": "30px",
                            "padding": "5px",
                        }
                    )
                ], style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    'position': 'relative',
                    "backgroundColor": ACCENT_COLOR,
                    "padding": "15px 30px",
                    "borderRadius": "30px",
                    "fontSize": "16px",
                    "color": TEXT_COLOR,
                    "flex": "1"
                }),
                html.Div(
                    ["😡" if pred == 0 else "😐" if pred == 1 else "😊"],
                    style={
                        "minWidth": "30px",
                        "textAlign": "center",
                        "fontSize": "30px",
                        "color": border_color,
                        "margin-left": "10px",
                        'padding': '5px 6.5px',
                        "borderRadius": "30px",
                        'border': f"1px solid {BORDER_COLOR}",
                    }
                )
            ], style={
                "display": "flex",
                "alignItems": "center",
                "gap": "10px",
                "borderLeft": f"5px solid {border_color}",
                "padding": "10px",
                "marginBottom": "10px"
            })
        )

    return messages


def chat_fine_tuning(model, tokenizer, text, pred, feedback):
    print('Типа дообучилась')
