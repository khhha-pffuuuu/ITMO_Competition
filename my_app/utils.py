import os

import base64
import io
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

import torch as t
from torch import nn
import torch.nn.functional as f

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TrainingArguments, Trainer, EarlyStoppingCallback)

from dash import html
from config import MAIN_BG_COLOR, BORDER_COLOR, CARD_BG_COLOR, TEXT_COLOR, ACCENT_COLOR, PLACEHOLDER_COLOR, BSM_PATH, METRIC_THR
from db.db_utils import get_user

from peft import get_peft_model, LoraConfig, PeftModel
from datasets import Dataset

import threading

from sklearn.metrics import recall_score

import wandb


# Загрузка модели
def load_model(user_model_path):
    tokenizer = AutoTokenizer.from_pretrained(BSM_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(BSM_PATH)

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
    tokenizer = AutoTokenizer.from_pretrained(BSM_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BSM_PATH)

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


def set_adapter(model):
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


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


def start_fine_tuning(model, tokenizer, user_id, train_data):
    thread = threading.Thread(target=fine_tuning, args=(model, tokenizer, user_id, train_data))
    thread.start()


def fine_tuning(model, tokenizer, user_id, train_data):
    print(f'Дообучение модели для пользователя {user_id}...')

    set_adapter(model)

    try:
        # Получим путь к модели пользователя
        user = get_user(user_id)
        user_model_path = user[2]

        eval_data = pd.read_excel('../data/test_dataset.xlsx')

        def preprocess_function(examples):
            inputs = tokenizer(
                examples['MessageText'],
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors="pt"
            )

            return inputs

        train_dataset = Dataset.from_pandas(train_data).map(preprocess_function, batched=True, num_proc=4)
        eval_dataset = Dataset.from_pandas(eval_data).map(preprocess_function, batched=True, num_proc=4)

        # Отключаем wandb
        wandb.init(mode="disabled")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)

            score = recall_score(labels, predictions, average="macro")

            return {"eval_recall": score}

        training_args = TrainingArguments(
            output_dir=f"{user_model_path}/tmp_finetune",
            run_name='output_dir',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            dataloader_num_workers=4,
            num_train_epochs=5,
            weight_decay=0.05,
            learning_rate=1e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_recall",
            greater_is_better=True,
            label_names=['labels'],
            report_to=None,
            disable_tqdm=True,
            use_cpu=True
        )

        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        trainer.train()

        eval_results = trainer.evaluate()
        eval_recall = eval_results.get('eval_recall', None)

        print(f'Значение Recall на тестовой выборке: {eval_recall}')
        if eval_recall is not None and eval_recall >= METRIC_THR:
            model.save_pretrained(user_model_path)
            print(f'Дообученная модель для пользователя {user_id} сохранена!')
        else:
            print(f'Значение целевой метрики оказалось ниже требуемого порога, модель не сохранена.')

    except Exception as e:
        print(f'Ошибка при дообучении: {e}')
