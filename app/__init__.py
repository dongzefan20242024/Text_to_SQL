from flask import Flask


def create_app():
    app = Flask(__name__, template_folder="../templates")
    app.config['SECRET_KEY'] = 'your_secret_key'

    # 注册蓝图
    with app.app_context():
        from .routes import main
        app.register_blueprint(main)

    return app
