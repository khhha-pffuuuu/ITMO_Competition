from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

import torch as t
from torch import nn
import torch.nn.functional as f

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    text = text.replace("\xa0", " ")

    return text.strip()


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")
    model = AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")

    model.classifier = nn.Sequential(
        nn.Linear(312, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Linear(512, 3)
    )

    model.to('cpu')
    model.load_state_dict(t.load("models/model.pth"))
    return tokenizer, model


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

    df['labels'] = pred

    idx2classes = {
        0: 'B',
        1: 'N',
        2: 'G'
    }
    df.labels = df.labels.apply(lambda x: idx2classes[x])

    unique_classes, counts = np.unique(df.labels, return_counts=True)
    stats = {i: 0 for i in ['B', 'N', 'G']}
    for i in range(len(unique_classes)):
        stats[unique_classes[i]] = counts[i]

    return df, stats
