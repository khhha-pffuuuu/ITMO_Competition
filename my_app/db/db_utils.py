import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "db.sqlite")


def add_user(user_id, model_path):
    """Добавляет нового пользователя, если его нет в БД"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (user_id, model_path) VALUES (?, ?)', (user_id, model_path))
        conn.commit()
    except sqlite3.IntegrityError:
        # Если пользователь уже существует, можно проигнорировать
        pass
    finally:
        conn.close()


def get_user(user_id):
    """Возвращает запись пользователя из БД"""
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))

    user = cursor.fetchone()
    conn.close()

    return user


def add_feedback(user_id, text, feedback):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (user_id, text, feedback, timestamp) VALUES (?, ?, ?, datetime('now'))",
            (user_id, text, feedback)
        )
        conn.commit()
