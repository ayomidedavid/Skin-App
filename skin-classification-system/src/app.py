from flask import Flask
from auth.routes import auth_bp
from dashboard.routes import dashboard_bp
from results.routes import results_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

app.register_blueprint(auth_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(results_bp)

if __name__ == '__main__':
    app.run(debug=True)