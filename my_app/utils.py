import base64
import io
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

import torch as t
from torch import nn
import torch.nn.functional as f

from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Загрузка модели
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path + '/tokenizer')
    model = AutoModelForSequenceClassification.from_pretrained(model_path + '/model')

    # model.classifier = nn.Sequential(
    #     nn.Linear(312, 4096),
    #     nn.ReLU(),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(),
    #     nn.Linear(4096, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 3)
    # )
    #
    # model.to('cpu')
    # model.load_state_dict(t.load(weights_path))

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
