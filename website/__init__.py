from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
import tkinter as tk
import threading
import time

db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    db.init_app(app)

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')

    from .models import User
    
    with app.app_context():
        db.create_all()

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Start Tkinter GUI in the background
    start_tkinter_in_background()

    return app

def start_tkinter_in_background():
    """Starts the Tkinter GUI in a background thread."""
    threading.Thread(target=start_tkinter, daemon=True).start()

def start_tkinter():
    """Initialize and start the Tkinter GUI."""
    root = tk.Tk()
    root.title("Tkinter GUI with Flask")

    # Label to display updates
    label = tk.Label(root, text="Starting...")
    label.pack(pady=20)

    # Background task to update the label
    def background_task():
        for i in range(10):
            time.sleep(1)  # Simulate work
            root.after(0, update_label, f"Count: {i+1}")  # Update Tkinter label safely

    def update_label(text):
        """Update Tkinter label."""
    label.config(text=text)

    # Start background task
    threading.Thread(target=background_task, daemon=True).start()

    # Start Tkinter main loop
    root.mainloop()

def create_database(app):
    """Creates the database if it doesn't exist."""
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')
