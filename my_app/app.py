import dash
from config import external_stylesheets
from layout import serve_layout
from callbacks import register_callbacks
from server import create_server

# Создаем Flask-сервер с маршрутом /init_user
server = create_server()

# Инициализация Dash-приложения
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)

app.title = "Анализ тональности"

# Установка layout
app.layout = serve_layout()

# Регистрация всех callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False, threaded=True)
