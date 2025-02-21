import os

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
from config import MAIN_BG_COLOR, BORDER_COLOR, CARD_BG_COLOR, TEXT_COLOR, ACCENT_COLOR, PLACEHOLDER_COLOR, base_model_path
from db.db_utils import get_user

from peft import get_peft_model, LoraConfig, PeftModel


# Загрузка модели
def load_model(user_model_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_path)

    model = PeftModel.from_pretrained(base_model, user_model_path)

    return tokenizer, model


def get_user_model(user_id):
    """Получает путь к модели пользователя из БД и загружает модель"""
    user = get_user(user_id)
    user_model_path = user[2]

    return load_model(user_model_path)


def create_peft_copy(user_model_path):
    """
    Создает PEFT‑копию базовой модели для нового пользователя.
    Если необходимо, здесь можно добавить вызов get_peft_model для дообучения модели.
    Пока что функция просто копирует базовую модель в новую папку.
    """
    # Создаем папку, если её нет
    if not os.path.exists(user_model_path):
        os.makedirs(user_model_path)

    # Загружаем базовую модель и токенизатор
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path)

    # Если хотите добавить PEFT, можно использовать примерно так (пример, не обязательный):
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['query', 'key', 'value'],
        bias="none"
    )
    model = get_peft_model(model, config)

    # Сохраняем модель в новую папку
    model.save_pretrained(user_model_path)


# Очистка текстов
def html_to_text(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text()
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


def parse_contents(contents, filename):
    """
    Декодирует файл, загружает его в pandas. DataFrame и выполняет предсказание.
    Обновляет глобальную переменную classes_counts.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xlsx' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None

        return df
    except Exception as e:
        print("Ошибка при чтении файла:", e)
        return None


def compute_stats(df):
    # Считаем статистику
    unique_classes, counts = np.unique(df.Class, return_counts=True)
    stats = {i: 0 for i in ['Negative', 'Neutral', 'Positive']}
    for i in range(len(unique_classes)):
        stats[unique_classes[i]] = counts[i]

    return stats


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
