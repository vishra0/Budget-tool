from flask import Blueprint, render_template, request, flash, jsonify, redirect
from flask_login import login_required, current_user
from . import db
from .models import Transaction,Account,Budget,User
from sqlalchemy import func,and_
from datetime import datetime, timedelta, timezone
import pytesseract
from werkzeug.utils import secure_filename
import os
import re
from transformers import pipeline
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO
import base64
import matplotlib.dates as mdates





views = Blueprint('views', __name__)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
categories = [
    "Food and Dining", "Personal Care and Grooming", "Technology and Electronics",
    "Clothing and Apparel", "Entertainment and Leisure", "Health and Wellness",
    "Transportation", "Housing and Living", "Education and Learning",
]


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        category = request.form.get('category')
        time = request.form.get('time')
        amount = request.form.get('amount')
        account_id = request.form.get('account_id')

        # Validate inputs
        if not category or len(category.strip()) == 0:
            flash('Category is required!', category='error')
        elif not amount or float(amount) <= 0:
            flash('Amount must be a positive number!', category='error')
        elif not account_id:
            flash('Account is required!', category='error')
        else:
            # Retrieve account and deduct amount
            account = Account.query.get(account_id)
            if account and account.user_id == current_user.id:
                if account.balance < float(amount):
                    flash('Insufficient funds in selected account!', category='error')
                else:
                    account.balance -= float(amount)
                    new_transaction = Transaction(
                        category=category,
                        amount=float(amount),
                        date=datetime.fromisoformat(time),
                        user_id=current_user.id,
                        account_id=account_id
                    )
                    db.session.add(new_transaction)
                    db.session.commit()
                    flash('Transaction created successfully!', category='success')
            else:
                flash('Invalid account selected!', category='error')

    # Render home.html with the categories
    return render_template("home.html", user=current_user, categories=categories)





@views.route('/delete-transaction', methods=['POST'])
@login_required
def delete_transaction():  
    data = json.loads(request.data)
    transaction_id = data['transactionId']
    transaction = Transaction.query.get(transaction_id)
    if transaction and transaction.user_id == current_user.id:
        db.session.delete(transaction)
        db.session.commit()
        flash('Transaction deleted!', category='success')
    return jsonify({})

@views.route('/accounts', methods=['GET', 'POST'])
@login_required
def accounts():
    if request.method == 'POST':
        name = request.form.get('name')
        balance = request.form.get('balance')

        if not name or len(name.strip()) == 0:
            flash('Account name is required!', category='error')
        elif not balance or float(balance) < 0:
            flash('Balance must be a non-negative number!', category='error')
        else:
            new_account = Account(name=name, balance=float(balance), user_id=current_user.id)
            db.session.add(new_account)
            db.session.commit()
            flash('Account created successfully!', category='success')

    return render_template("accounts.html", user=current_user)

@views.route('/budgets', methods=['GET'])
@login_required
def budgets():
    """Display budgets and update them before rendering."""
    update_budget()  # Ensure spent amounts are up-to-date
    return render_template("budgets.html", user=current_user)


@views.route('/add_budget', methods=['POST'])
@login_required
def add_budget():
    """Add a new budget for a specific period."""
    period = request.form.get('period')
    target_amount = request.form.get('target_amount')

    if not target_amount or float(target_amount) <= 0:
        flash('Target amount must be greater than zero!', category='error')
    else:
        # Check if the user already has a budget for this period
        existing_budget = Budget.query.filter_by(user_id=current_user.id, period=period).first()
        if existing_budget:
            # Update the existing budget's target amount
            existing_budget.target_amount = float(target_amount)
            flash(f'{period.capitalize()} budget updated successfully!', category='success')
        else:
            # Create a new budget if none exists
            new_budget = Budget(
                period=period,
                target_amount=float(target_amount),
                spent_amount=0,  # Initialize spent amount as 0
                user_id=current_user.id
            )
            db.session.add(new_budget)
            flash(f'{period.capitalize()} budget created successfully!', category='success')

        db.session.commit()

    return redirect('/budgets')


@views.route('/update_budget', methods=['POST', 'GET'])
@login_required
def update_budget():
    """Update spent amounts for all budgets based on transaction data."""
    periods = ['weekly', 'monthly', 'yearly']

    # Define start dates for each budget period
    today = datetime.now(timezone.utc)
    start_dates = {
        'weekly': today - timedelta(days=today.weekday()),  # Start of the current week (Monday)
        'monthly': today.replace(day=1),                   # Start of the current month
        'yearly': today.replace(month=1, day=1),           # Start of the current year
    }

    for period in periods:
        start_date = start_dates[period]
        end_date = today

        # Calculate the total amount spent in this period
        total_spent = db.session.query(func.sum(Transaction.amount)).filter(
            and_(
                Transaction.date >= start_date,
                Transaction.date <= end_date,
                Transaction.user_id == current_user.id
            )
        ).scalar() or 0  

        # Find the budget for this period
        budget = Budget.query.filter_by(user_id=current_user.id, period=period).first()

        if budget:
           
            budget.spent_amount = total_spent
        else:
          
            continue

    # Commit all updates
    db.session.commit()
    flash('Budgets updated successfully!', category='success')

    return redirect('/budgets')


@views.route('/upload_receipt', methods=['POST'])
@login_required
def upload_receipt():
    if 'receipt_image' not in request.files:
        flash('No file part', category='error')
        return jsonify({"success": False, "message": "No file part"}), 400

    file = request.files['receipt_image']
    if file.filename == '':
        flash('No selected file', category='error')
        return jsonify({"success": False, "message": "No selected file"}), 400

    # Save file temporarily
    filepath = os.path.join('uploads', secure_filename(file.filename))
    file.save(filepath)

    try:
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(filepath)

        # Regex patterns for UPI, Google Pay-specific details, and amounts
        upi_pattern = r"UPI Transaction ID[:\s]+([\d]+)"
        google_pay_pattern = r"Google Transaction ID[:\s]+([\w\-]+)"
        amount_context_pattern = r"(?:Paid|Total|Amount|₹|\$)\s?[:₹$]?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)"

        # Detect UPI transaction ID
        upi_match = re.search(upi_pattern, extracted_text)
        upi_transaction_id = upi_match.group(1) if upi_match else None

        # Detect Google Transaction ID
        google_match = re.search(google_pay_pattern, extracted_text)
        google_transaction_id = google_match.group(1) if google_match else None

        # Extract amount based on context
        amount_matches = re.findall(amount_context_pattern, extracted_text)

        # Normalize amounts (remove commas) and convert to float
        normalized_amounts = [float(amount.replace(',', '')) for amount in amount_matches]

        # Determine the most likely amount
        total_amount = max(normalized_amounts) if normalized_amounts else 0

        if total_amount <= 0:
            return jsonify({"success": False, "message": "Could not detect a valid total amount"}), 400

        # Use Hugging Face Transformers to predict the category
        result = classifier(extracted_text, categories)
        detected_category = result['labels'][0]  # Top predicted category

        # Construct the response
        response = {
            "success": True,
            "category": detected_category,
            "amount": total_amount,
        }

        if upi_transaction_id:
            response['upi_transaction_id'] = upi_transaction_id
        if google_transaction_id:
            response['google_transaction_id'] = google_transaction_id

        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

    finally:
        # Clean up the temporary file
        if os.path.exists(filepath):
            os.remove(filepath)




@views.route('/add_transaction', methods=['POST'])
@login_required
def add_transaction():
    category = request.form.get('category')
    time = request.form.get('time')
    amount = request.form.get('amount')
    account_id = request.form.get('account_id')

    # Validate inputs
    if not category or len(category.strip()) == 0:
        flash('Category is required!', category='error')
        return redirect('/')
    elif not amount or float(amount) <= 0:
        flash('Amount must be a positive number!', category='error')
        return redirect('/')
    elif not account_id:
        flash('Account is required!', category='error')
        return redirect('/')

    try:
        # Retrieve the account
        account = Account.query.get(account_id)
        if account and account.user_id == current_user.id:
            if account.balance < float(amount):
                flash('Insufficient funds in selected account!', category='error')
            else:
                account.balance -= float(amount)
                new_transaction = Transaction(
                    category=category,
                    amount=float(amount),
                    date=datetime.now(),  # or datetime.fromisoformat(time) if provided
                    user_id=current_user.id,
                    account_id=account_id
                )
                db.session.add(new_transaction)
                db.session.commit()
                flash('Transaction added successfully!', category='success')
        else:
            flash('Invalid account selected!', category='error')
    except Exception as e:
        flash(f"Error adding transaction: {str(e)}", category='error')

    return redirect('/')

@views.route('/projections', methods=['GET'])
@login_required
def projections():
    # Fetch transaction data
    transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date).all()

    # Prepare data for regression
    dates = np.array([(t.date - datetime(1970, 1, 1)).days for t in transactions]).reshape(-1, 1)
    amounts = np.array([t.amount for t in transactions]).reshape(-1, 1)

    # Train the model
    model = LinearRegression()
    model.fit(dates, amounts)

    # Create future dates for prediction (next 12 months, 1 month at a time)
    future_dates = np.array([dates[-1][0] + i * 30 for i in range(1, 13)]).reshape(-1, 1)  # Predict for next 12 months
    future_amounts = model.predict(future_dates)

    # Convert days to datetime objects
    transaction_dates = [datetime(1970, 1, 1) + timedelta(days=int(d)) for d in dates.flatten()]
    future_dates_dt = [datetime(1970, 1, 1) + timedelta(days=int(d)) for d in future_dates.flatten()]

    # Generate graph for total projections
    plt.figure(figsize=(10, 6))
    plt.plot(transaction_dates, amounts, label='Actual Transactions', marker='o')
    plt.plot(future_dates_dt, future_amounts, label='Projected Transactions', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Amount Spent')
    plt.title('Total Transaction Projections')
    plt.legend()
    plt.grid(True)

    # Format the x-axis to show months
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Major ticks at months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as 'Jan 2024', 'Feb 2024'
    plt.xticks(rotation=45)

    # Save graph to a base64 string
    total_graph = BytesIO()
    plt.savefig(total_graph, format="png")
    total_graph.seek(0)
    total_graph_base64 = base64.b64encode(total_graph.read()).decode('utf-8')
    plt.close()

    # Generate graphs for categories
    category_graphs = {}
    category_data = {}
    for category in set(t.category for t in transactions):
        cat_transactions = [t for t in transactions if t.category == category]
        cat_dates = np.array([(t.date - datetime(1970, 1, 1)).days for t in cat_transactions]).reshape(-1, 1)
        cat_amounts = np.array([t.amount for t in cat_transactions]).reshape(-1, 1)
        if len(cat_dates) > 1:  # Ensure sufficient data for regression
            cat_model = LinearRegression()
            cat_model.fit(cat_dates, cat_amounts)
            cat_future_amounts = cat_model.predict(future_dates)

            # Convert days to datetime objects for categories
            cat_transaction_dates = [datetime(1970, 1, 1) + timedelta(days=int(d)) for d in cat_dates.flatten()]
            category_data[category] = (cat_transaction_dates, cat_amounts, cat_future_amounts)

    # Generate category graphs
    for category, (cat_dates, cat_amounts, cat_future_amounts) in category_data.items():
        plt.figure(figsize=(10, 6))
        plt.plot(cat_dates, cat_amounts, label=f'{category} - Actual', marker='o')
        plt.plot(future_dates_dt, cat_future_amounts, label=f'{category} - Projected', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Amount Spent')
        plt.title(f'{category} Projections')
        plt.legend()
        plt.grid(True)

        # Format x-axis for category graphs
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Major ticks at months
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as 'Jan 2024', 'Feb 2024'
        plt.xticks(rotation=45)

        cat_graph = BytesIO()
        plt.savefig(cat_graph, format="png")
        cat_graph.seek(0)
        category_graphs[category] = base64.b64encode(cat_graph.read()).decode('utf-8')
        plt.close()

    return render_template(
        "projections.html",
        user=current_user,
        total_graph=total_graph_base64,
        category_graphs=category_graphs
    )