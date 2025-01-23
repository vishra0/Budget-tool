from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func
from datetime import datetime, timedelta, timezone
from sqlalchemy import Enum

# Transaction Model

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime(timezone=True), default=datetime.now(timezone.utc))
    account_id = db.Column(db.Integer, db.ForeignKey('account.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)



# Predefined categories
CATEGORIES = [
    "Food and Dining", "Personal Care and Grooming", "Technology and Electronics",
    "Clothing and Apparel", "Entertainment and Leisure", "Health and Wellness",
    "Transportation", "Housing and Living", "Education and Learning"
]


# Budget Model
class Budget(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    period = db.Column(db.String(50), nullable=False)  # e.g., "monthly", "weekly"
    target_amount = db.Column(db.Float, nullable=False)
    spent_amount = db.Column(db.Float, default=0)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

# Account Model
class Account(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    balance = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    transactions = db.relationship('Transaction', backref='account', lazy=True)

# User Model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    transactions = db.relationship('Transaction', backref='user', lazy=True)
    budgets = db.relationship('Budget', backref='user', lazy=True)
    accounts = db.relationship('Account', backref='user', lazy=True)
