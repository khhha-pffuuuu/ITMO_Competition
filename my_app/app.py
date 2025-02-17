import dash
from config import external_stylesheets, model_path
from layout import serve_layout
from callbacks import register_callbacks
from utils import load_model

# Инициализация Dash-приложения
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Анализ тональности"

# Загрузка модели
tokenizer, model = load_model(model_path)

# Установка layout
app.layout = serve_layout()

# Регистрация всех callbacks
register_callbacks(app, model=model, tokenizer=tokenizer)

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False, threaded=True)
