from flask import Flask, request, make_response, redirect
import uuid

from utils import create_peft_copy

from db.db_setup import initialize_db
from db.db_utils import add_user, get_user


def create_server():
    server = Flask(__name__)
    initialize_db()

    @server.route("/init_user")
    def init_user():
        user_id = request.cookies.get("user_id")
        if not user_id:
            user_id = str(uuid.uuid4())

        if not get_user(user_id):
            user_model_path = f"../models/{user_id}"

            create_peft_copy(user_model_path)

            add_user(user_id, model_path=user_model_path)
            response = make_response(redirect("/"))
            response.set_cookie("user_id", user_id, max_age=365 * 30 * 24 * 60 * 60)  # Файлы cookie хранятся год

            return response

        return "", 204

    return server


if __name__ == "__main__":
    app = create_server()
    app.run(debug=True)
