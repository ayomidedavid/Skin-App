from flask import Blueprint, request, render_template, redirect, url_for, flash, session
from .utils import hash_password, check_password, is_user_exists
from db.connection import get_db_connection

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if is_user_exists(username):
            flash('Username already exists. Please choose a different one.')
            return redirect(url_for('auth.register'))
        
        hashed_password = hash_password(password)
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)", (username, email, hashed_password))
        connection.commit()
        cursor.close()
        connection.close()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        connection.close()
        
        if user and check_password(user['password_hash'], password):
            session['username'] = username
            flash('Login successful!')
            return redirect(url_for('dashboard.index'))
        else:
            flash('Invalid username or password. Please try again.')
    
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('auth.login'))

@auth_bp.errorhandler(405)
def method_not_allowed(e):
    return render_template('405.html'), 405