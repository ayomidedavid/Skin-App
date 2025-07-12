import bcrypt
import base64

def hash_password(password):
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    return base64.b64encode(hashed).decode('utf-8')

def check_password(hashed_password, user_password):
    hashed_password_bytes = base64.b64decode(hashed_password.encode('utf-8'))
    return bcrypt.checkpw(user_password.encode('utf-8'), hashed_password_bytes)

def validate_user_input(username, password):
    if not username or not password:
        return False
    return True

def create_session(user_id):
    from flask import session
    session['user_id'] = user_id

def logout():
    from flask import session
    session.pop('user_id', None)

def is_user_exists(username):
    from db.connection import execute_query
    query = "SELECT id FROM users WHERE username = %s"
    result = execute_query(query, (username,))
    return len(result) > 0