import sqlite3
import os
import pandas as pd

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


def extract_and_delete_feedback(user_id):
    """
    Если для пользователя user_id накопилось не менее 100 записей в таблице feedback,
    извлекает 100 самых старых записей (по timestamp) в DataFrame и удаляет их из базы.
    Если записей меньше 100, возвращает None.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Получаем количество записей для user_id
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE user_id = ?", (user_id,))
        count = cursor.fetchone()[0]

        if count < 100:
            return None

        print(f'Набрано 100 сообщений для пользователя {user_id}')

        # Извлекаем 100 старейших записей
        cursor.execute("""
            SELECT id, user_id, text, feedback, timestamp
            FROM feedback
            WHERE user_id = ?
            ORDER BY timestamp ASC
            LIMIT 100
        """, (user_id,))
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=["id", "user_id", "MessageText", "labels", "timestamp"])
        df.labels = df.labels.astype(int)

        # Удаляем извлечённые записи
        ids = df["id"].tolist()
        placeholders = ",".join("?" for _ in ids)
        cursor.execute(f"DELETE FROM feedback WHERE id IN ({placeholders})", tuple(ids))
        conn.commit()

        return df
