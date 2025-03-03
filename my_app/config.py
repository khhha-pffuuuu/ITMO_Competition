import dash_bootstrap_components as dbc

# Внешние стили (темы Bootstrap)
STYLESHEETS = [dbc.themes.DARKLY]

# Цветовая схема
MAIN_BG_COLOR = "#222222"
BORDER_COLOR = "#505050"
CARD_BG_COLOR = "#303030"
TEXT_COLOR = "#E0E0E0"
ACCENT_COLOR = "#404040"
PLACEHOLDER_COLOR = "#888888"

# Название базовой модели и путь к весам дообученной
BSM_PATH = "../models/base_model"

# Словарь вида: Id -> Тональность
ID2CLS = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

# Порог для сохранения дообученной модели
METRIC_THR = 0.7
