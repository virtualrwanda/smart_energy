import os
import sqlite3
import datetime
import logging
import secrets
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_wtf.csrf import CSRFProtect, generate_csrf
import pandas as pd
import numpy as np
import requests
from werkzeug.security import generate_password_hash, check_password_hash
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'energy_meter.db')
    SMS_API_URL = os.getenv('SMS_API_URL', 'https://vrt.rw/SMS/sms.php')
    CONVERSION_RATE = float(os.getenv('CONVERSION_RATE', '0.0047'))
    MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'optimized_customer_payment_model.pkl')
    WTF_CSRF_ENABLED = True
    WTF_CSRF_SECRET_KEY = os.getenv('CSRF_SECRET_KEY', secrets.token_hex(32))
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour

# Apply configuration
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['WTF_CSRF_ENABLED'] = Config.WTF_CSRF_ENABLED
app.config['WTF_CSRF_SECRET_KEY'] = Config.WTF_CSRF_SECRET_KEY
app.config['WTF_CSRF_TIME_LIMIT'] = Config.WTF_CSRF_TIME_LIMIT

# Initialize CSRF protection
csrf = CSRFProtect(app)

# ==================== MODEL LOADING WITH ENHANCED ERROR HANDLING ====================
def load_machine_learning_model():
    """Load ML model with multiple fallback options"""
    import joblib
    import pickle
    
    # Possible model filenames to try
    model_filenames = [
        Config.MODEL_FILENAME,
        'cfoptimized_customer_payment_model.pkl',
        'optimized_customer_payment_model.pkl',
        'foptimized_customer_payment_model.pkl',
        'model.pkl'
    ]
    
    # Get absolute path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    for filename in model_filenames:
        model_path = os.path.join(base_dir, filename)
        
        if os.path.exists(model_path):
            logger.info(f"Found model file: {model_path}")
            logger.info(f"File size: {os.path.getsize(model_path)} bytes")
            
            # Try different loading methods
            loading_methods = [
                ('joblib', lambda: joblib.load(model_path)),
                ('pickle', lambda: pickle.load(open(model_path, 'rb'))),
                ('joblib_compat', lambda: joblib.load(model_path, mmap_mode='r'))
            ]
            
            for method_name, load_func in loading_methods:
                try:
                    logger.info(f"Attempting to load with {method_name}...")
                    model = load_func()
                    
                    # Test the model with sample data
                    test_input = np.array([[5000, 4000]])
                    if hasattr(model, 'predict'):
                        test_prediction = model.predict(test_input)
                        logger.info(f"✓ Model loaded successfully with {method_name}!")
                        logger.info(f"  Model type: {type(model).__name__}")
                        logger.info(f"  Test prediction: {test_prediction[0]:.2f}")
                        
                        # Check model features
                        if hasattr(model, 'n_features_in_'):
                            logger.info(f"  Expected features: {model.n_features_in_}")
                        if hasattr(model, 'feature_names_in_'):
                            logger.info(f"  Feature names: {model.feature_names_in_}")
                        
                        return model
                    else:
                        logger.warning(f"Loaded object has no predict method: {type(model)}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load with {method_name}: {str(e)}")
                    continue
    
    # If no model found, create a fallback model
    logger.warning("No valid model file found. Creating fallback model...")
    return create_fallback_model()

def create_fallback_model():
    """Create a simple fallback model for predictions"""
    try:
        from sklearn.linear_model import LinearRegression
        
        # Create training data based on typical payment patterns
        # Pattern: next_payment = 0.7 * previous + 0.3 * two_months_ago
        X_train = np.array([
            [1000, 800], [1500, 1200], [2000, 1800], [2500, 2200],
            [3000, 2800], [3500, 3200], [4000, 3800], [4500, 4200],
            [5000, 4800], [5500, 5200], [6000, 5800], [7000, 6500],
            [8000, 7500], [9000, 8500], [10000, 9500]
        ])
        
        # Target: weighted average with slight randomness
        y_train = X_train[:, 0] * 0.7 + X_train[:, 1] * 0.3 + np.random.randn(len(X_train)) * 50
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        logger.info("✓ Fallback model created successfully")
        logger.info(f"  Model R² score: {model.score(X_train, y_train):.3f}")
        
        # Save the fallback model for future use
        import joblib
        model_path = os.path.join(os.path.dirname(__file__), 'cfoptimized_customer_payment_model.pkl')
        joblib.dump(model, model_path)
        logger.info(f"  Fallback model saved to {model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to create fallback model: {e}")
        logger.error(traceback.format_exc())
        return None

# Load the model
logger.info("=" * 50)
logger.info("LOADING MACHINE LEARNING MODEL")
logger.info("=" * 50)
loaded_model = load_machine_learning_model()

if loaded_model:
    logger.info("✅ Model is ready for predictions")
else:
    logger.warning("⚠️ No model available. Using simple weighted average fallback.")

# ==================== DATABASE FUNCTIONS ====================
def get_db_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with all required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create meter_data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS meter_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            serial_number TEXT NOT NULL UNIQUE,
            balance REAL NOT NULL DEFAULT 0,
            consumption REAL NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL,
            owner_name TEXT NOT NULL,
            owner_contact TEXT NOT NULL
        )
    ''')

    # Create alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')

    # Create recharges table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recharges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            serial_number TEXT NOT NULL,
            recharge_amount REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    
    # Create purchase table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS purchase (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            serial_number TEXT NOT NULL,
            recharge_amount REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    
    # Create users table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE,
                role TEXT DEFAULT 'user',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert default admin user
        admin_password = generate_password_hash('admin123')
        cursor.execute('''
            INSERT INTO users (username, password_hash, email, role)
            VALUES (?, ?, ?, ?)
        ''', ('admin', admin_password, 'admin@example.com', 'admin'))
        
        # Insert sample client user
        client_password = generate_password_hash('client123')
        cursor.execute('''
            INSERT INTO users (username, password_hash, email, role)
            VALUES (?, ?, ?, ?)
        ''', ('client', client_password, 'client@example.com', 'client'))
    else:
        # Check and fix schema if needed
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'password' in columns and 'password_hash' not in columns:
            cursor.execute("DROP TABLE users")
            cursor.execute('''
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT UNIQUE,
                    role TEXT DEFAULT 'user',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            admin_password = generate_password_hash('admin123')
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
            ''', ('admin', admin_password, 'admin@example.com', 'admin'))
            
            client_password = generate_password_hash('client123')
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
            ''', ('client', client_password, 'client@example.com', 'client'))
    
    # Add sample meter for client
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO meter_data (serial_number, balance, consumption, timestamp, owner_name, owner_contact)
            VALUES (?, 50.0, 0, datetime("now"), ?, ?)
        ''', ('08928382783', 'client', '0788123456'))
    except Exception as e:
        logger.error(f"Error adding sample meter: {e}")
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_meter_serial ON meter_data(serial_number)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_recharges_serial ON recharges(serial_number)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_recharges_timestamp ON recharges(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# ==================== USER CLASS ====================
class User:
    def __init__(self, id, username, role, serial_number=None):
        self.id = id
        self.username = username
        self.role = role
        self.serial_number = serial_number
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
    
    def get_id(self):
        return str(self.id)
    
    @property
    def is_admin(self):
        return self.role == 'admin'
    
    @property
    def is_client(self):
        return self.role == 'client'

# ==================== CONTEXT PROCESSORS ====================
@app.context_processor
def inject_user():
    """Make current_user available in all templates"""
    if 'user_id' in session:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.id, u.username, u.role, m.serial_number 
            FROM users u 
            LEFT JOIN meter_data m ON u.username = m.owner_name
            WHERE u.id = ?
        ''', (session['user_id'],))
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data:
            user = User(
                user_data['id'], 
                user_data['username'], 
                user_data['role'],
                user_data['serial_number']
            )
            return dict(current_user=user)
    
    class AnonymousUser:
        is_authenticated = False
        is_anonymous = True
        role = 'guest'
        is_admin = False
        is_client = False
        serial_number = None
    
    return dict(current_user=AnonymousUser())

@app.context_processor
def inject_config():
    """Make config variables available to all templates"""
    return {
        'config': {
            'CONVERSION_RATE': Config.CONVERSION_RATE,
            'SMS_API_URL': Config.SMS_API_URL,
            'DATABASE_PATH': Config.DATABASE_PATH,
            'MODEL_FILENAME': Config.MODEL_FILENAME
        },
        'loaded_model': loaded_model is not None
    }

# ==================== DECORATORS ====================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def api_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({
                'success': False,
                'error': 'Authentication required. Please login first.'
            }), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin':
            flash('Access denied. Admin only.', 'error')
            return redirect(url_for('client_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== HELPER FUNCTIONS ====================
def error_response(message, status_code=400):
    """Standardized error response"""
    return jsonify({
        'success': False,
        'error': message,
        'timestamp': datetime.datetime.now().isoformat()
    }), status_code

def success_response(data=None, message="Success", status_code=200):
    """Standardized success response"""
    response = {
        'success': True,
        'message': message,
        'timestamp': datetime.datetime.now().isoformat()
    }
    if data is not None:
        response['data'] = data
    return jsonify(response), status_code

def send_sms(phone, message):
    """Send an SMS alert when balance is low"""
    try:
        url = f"{Config.SMS_API_URL}?phone={phone}&message={message}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            logger.info(f"SMS sent successfully to {phone}")
            return True
        else:
            logger.error(f"SMS failed with status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"SMS Sending Error: {e}")
        return False

def predict_with_fallback(prev_payment, two_months_payment):
    """Make prediction using model or fallback method"""
    if loaded_model is not None:
        try:
            # Prepare input based on model requirements
            if hasattr(loaded_model, 'n_features_in_'):
                if loaded_model.n_features_in_ == 2:
                    input_data = np.array([[float(prev_payment), float(two_months_payment)]])
                else:
                    # Handle different feature counts
                    input_data = np.random.rand(1, loaded_model.n_features_in_) * 10000
                    logger.warning(f"Model expects {loaded_model.n_features_in_} features, using default")
            else:
                input_data = np.array([[float(prev_payment), float(two_months_payment)]])
            
            prediction = loaded_model.predict(input_data)[0]
            logger.info(f"ML Prediction: {prediction:.2f}")
            return prediction
        except Exception as e:
            logger.error(f"ML Prediction failed: {e}")
            logger.info("Falling back to weighted average")
    
    # Fallback: weighted average (70% recent, 30% older)
    prediction = (float(prev_payment) * 0.7) + (float(two_months_payment) * 0.3)
    logger.info(f"Fallback Prediction: {prediction:.2f}")
    return prediction

# ==================== AUTHENTICATION ROUTES ====================
@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Handle user login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, password_hash, role FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            flash('Login successful!', 'success')
            
            if user['role'] == 'admin':
                return redirect(url_for('index'))
            else:
                return redirect(url_for('client_dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Handle user logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login_page'))

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    """Handle user registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if not all([username, password, email]):
            flash('All fields are required', 'error')
            return render_template('register.html')
        
        password_hash = generate_password_hash(password)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
            ''', (username, password_hash, email, 'client'))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login_page'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

# ==================== MAIN ROUTES ====================
@app.route('/')
@login_required
def index():
    """Render main dashboard (admin view)"""
    if session.get('role') == 'admin':
        return render_template('index.html')
    else:
        return redirect(url_for('client_dashboard'))

@app.route('/client/dashboard')
@login_required
def client_dashboard():
    """Render client dashboard"""
    return render_template('client_dashboard.html')

@app.route('/pp')
@login_required
def prediction_page():
    """Render prediction page"""
    # Get meters for admin dropdown
    meters = []
    if session.get('role') == 'admin':
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT serial_number, owner_name, balance FROM meter_data ORDER BY serial_number')
        meters = [dict(row) for row in cursor.fetchall()]
        conn.close()
    
    return render_template('prediction.html', meters=meters, loaded_model=loaded_model)

@app.route('/meters')
@login_required
@admin_required
def meters_page():
    """Render meters management page (admin only)"""
    return render_template('meters.html')

@app.route('/recharge')
@login_required
def recharge_page():
    """Render recharge page"""
    meters = []
    if session.get('role') == 'admin':
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT serial_number, owner_name, balance FROM meter_data ORDER BY serial_number')
        meters = [dict(row) for row in cursor.fetchall()]
        conn.close()
    
    return render_template('recharge.html', meters=meters)

@app.route('/users')
@login_required
@admin_required
def users_page():
    """Render users management page (admin only)"""
    return render_template('users.html')

# ==================== API ROUTES ====================
# @app.route('/api/get_csrf_token', methods=['GET'])
# def get_csrf_token():
#     """Return a CSRF token for AJAX requests"""
#     token = generate_csrf()
#     return jsonify({'csrf_token': token})
@app.route('/api/csrf_token', methods=['GET'])
def get_csrf_token():
    """Get CSRF token for API clients"""
    from flask_wtf.csrf import generate_csrf
    return success_response({
        'csrf_token': generate_csrf()
    }, "CSRF token generated")

@app.route('/api/check-auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated (returns JSON)"""
    if 'user_id' in session:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT username, role FROM users WHERE id = ?', (session['user_id'],))
        user = cursor.fetchone()
        conn.close()
        
        return jsonify({
            'success': True,
            'authenticated': True,
            'user_id': session['user_id'],
            'username': user['username'] if user else None,
            'role': user['role'] if user else None
        })
    else:
        return jsonify({
            'success': False,
            'authenticated': False,
            'message': 'Not authenticated'
        })

@app.route('/api/test', methods=['GET'])
def test_api():
    """Test endpoint to verify API is working"""
    return jsonify({
        'success': True,
        'message': 'API is working',
        'timestamp': datetime.datetime.now().isoformat(),
        'model_loaded': loaded_model is not None
    })

@app.route('/api/data', methods=['GET'])
@api_login_required
def get_data():
    """Get meter data for charts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT timestamp, consumption FROM meter_data ORDER BY timestamp DESC LIMIT 100')
        data = cursor.fetchall()
        conn.close()
        
        result = [(row['timestamp'], row['consumption']) for row in data]
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in get_data: {e}")
        return jsonify([])

@app.route('/api/registered_meters', methods=['GET'])
@api_login_required
def get_registered_meters():
    """Get all registered meters"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT serial_number, owner_name, owner_contact, balance, timestamp FROM meter_data ORDER BY timestamp DESC')
        meters = cursor.fetchall()
        conn.close()

        result = [dict(row) for row in meters]
        return success_response(result)
    except Exception as e:
        logger.error(f"Error in get_registered_meters: {e}")
        return error_response(str(e), 500)

@app.route('/api/register', methods=['POST'])
@api_login_required
@admin_required
def register():
    """Register a new meter"""
    try:
        if not request.is_json:
            return error_response("Content-Type must be application/json", 400)
            
        data = request.get_json()
        if not data:
            return error_response("No data provided", 400)
            
        serial_number = data.get('serial_number', '').strip()
        owner_name = data.get('owner_name', '').strip()
        owner_contact = data.get('owner_contact', '').strip()

        if not all([serial_number, owner_name, owner_contact]):
            return error_response("Missing required fields. Please provide serial_number, owner_name, and owner_contact.", 400)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT serial_number FROM meter_data WHERE serial_number = ?', (serial_number,))
        if cursor.fetchone():
            conn.close()
            return error_response(f"Meter with serial number {serial_number} already exists.", 400)
        
        cursor.execute('''
            INSERT INTO meter_data (serial_number, balance, consumption, timestamp, owner_name, owner_contact)
            VALUES (?, 0, 0, datetime("now"), ?, ?)
        ''', (serial_number, owner_name, owner_contact))
        
        conn.commit()
        conn.close()
        
        logger.info(f"New meter registered: {serial_number}")
        
        return success_response({
            'serial_number': serial_number,
            'owner_name': owner_name,
            'owner_contact': owner_contact
        }, "Meter registered successfully!")
        
    except sqlite3.IntegrityError:
        return error_response("Meter with this serial number already exists.", 400)
    except Exception as e:
        logger.error(f"Error registering meter: {e}")
        return error_response(str(e), 500)

@app.route('/api/recharge', methods=['POST'])
@api_login_required
def recharge():
    """Recharge a meter"""
    try:
        if not request.is_json:
            return error_response("Content-Type must be application/json", 400)
            
        data = request.get_json()
        
        if session.get('role') == 'admin':
            serial_number = data.get('meter_serial_number', '').strip()
            if not serial_number:
                return error_response("Please select a meter", 400)
        else:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT serial_number FROM meter_data 
                WHERE owner_name = (SELECT username FROM users WHERE id = ?)
            ''', (session['user_id'],))
            meter = cursor.fetchone()
            conn.close()
            
            if not meter:
                return error_response("No meter associated with your account", 400)
            
            serial_number = meter['serial_number']
        
        amount = data.get('recharge_amount', 0)

        try:
            amount = float(amount)
            credit_amount = amount * Config.CONVERSION_RATE
        except ValueError:
            return error_response("Invalid recharge amount.", 400)

        if not serial_number or credit_amount <= 0:
            return error_response("Invalid serial number or amount.", 400)

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT balance FROM meter_data WHERE serial_number = ?', (serial_number,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return error_response("Meter not found.", 404)
        
        current_balance = row['balance']
        new_balance = current_balance + credit_amount
        
        cursor.execute('UPDATE meter_data SET balance = ? WHERE serial_number = ?', 
                     (new_balance, serial_number))
        
        cursor.execute('''
            INSERT INTO recharges (serial_number, recharge_amount, timestamp)
            VALUES (?, ?, datetime("now"))
        ''', (serial_number, credit_amount))
        
        cursor.execute('''
            INSERT INTO purchase (serial_number, recharge_amount, timestamp)
            VALUES (?, ?, datetime("now"))
        ''', (serial_number, credit_amount))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Recharge successful for {serial_number}: {credit_amount}")
        
        return jsonify({
            'success': True,
            'message': 'Recharge successful!',
            'new_balance': new_balance,
            'credit_amount': credit_amount
        }), 200
            
    except Exception as e:
        logger.error(f"Error processing recharge: {e}")
        return error_response(str(e), 500)

@app.route('/api/recharges', methods=['GET'])
@api_login_required
def get_recharge_history():
    """Get recharge history for a meter"""
    serial_number = request.args.get('serial_number', '').strip()
    
    if not serial_number:
        return error_response("Serial number is required.", 400)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT recharge_amount, timestamp FROM recharges 
            WHERE serial_number = ?
            ORDER BY timestamp DESC
        ''', (serial_number,))
        recharges = cursor.fetchall()
        conn.close()

        result = [dict(row) for row in recharges]
        return success_response(result)
    except Exception as e:
        logger.error(f"Error in get_recharge_history: {e}")
        return error_response(str(e), 500)

@app.route('/api/get_recharges', methods=['GET'])
@api_login_required
def get_recharges():
    """Get all recharges for charts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT timestamp, recharge_amount FROM recharges ORDER BY timestamp DESC LIMIT 100')
        recharges = cursor.fetchall()
        conn.close()

        data = [{'timestamp': row['timestamp'], 'recharge_amount': row['recharge_amount']} for row in recharges]
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in get_recharges: {e}")
        return jsonify([])

@app.route('/api/meter_status', methods=['GET'])
#@api_login_required
def get_meter_status():
    """Get current status of a meter"""
    serial_number = request.args.get('serial_number', '').strip()
    
    if not serial_number:
        return error_response("Serial number is required.", 400)
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT balance, consumption FROM meter_data WHERE serial_number = ?', (serial_number,))
        data = cursor.fetchone()
        conn.close()
        
        if data:
            balance, consumption = data['balance'], data['consumption']
            status = "Low Balance" if balance <= 1 else "Normal"
            return success_response({
                "balance": balance, 
                "consumption": consumption, 
                "status": status
            })
        else:
            return error_response("Meter not found.", 404)
    except Exception as e:
        logger.error(f"Error in get_meter_status: {e}")
        return error_response(str(e), 500)

@app.route('/api/consume', methods=['POST'])
def consume():
    """Record energy consumption"""
    try:
        if not request.is_json:
            return error_response("Content-Type must be application/json", 400)
            
        data = request.get_json()
        serial_number = data.get('meter_serial_number', '').strip()
        consumption_amount = data.get('consumption_amount', 0)

        try:
            consumption_amount = float(consumption_amount)
        except ValueError:
            return error_response("Invalid consumption amount.", 400)

        if not serial_number or consumption_amount < 0:
            return error_response("Invalid serial number or consumption amount.", 400)

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT balance, owner_contact FROM meter_data WHERE serial_number = ?', (serial_number,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return error_response("Meter not found.", 404)

        current_balance = row['balance']
        owner_contact = row['owner_contact']

        if current_balance <= 0:
            conn.close()
            return error_response("Insufficient balance.", 400)

        new_balance = max(0, current_balance - consumption_amount)

        cursor.execute('UPDATE meter_data SET balance = ?, consumption = consumption + ? WHERE serial_number = ?', 
                      (new_balance, consumption_amount, serial_number))

        cursor.execute('''
            INSERT INTO recharges (serial_number, recharge_amount, timestamp)
            VALUES (?, ?, datetime("now"))
        ''', (serial_number, -consumption_amount))

        conn.commit()
        conn.close()

        sms_status = None
        if new_balance <= 1 and owner_contact:
            sms_message = f"Alert! Your energy meter balance is low: {new_balance:.2f} kWh. Please recharge soon."
            sms_status = send_sms(owner_contact, sms_message)

        response_data = {"new_balance": new_balance}
        if sms_status is not None:
            response_data["sms_sent"] = sms_status

        return success_response(response_data, "Consumption recorded successfully!")

    except Exception as e:
        logger.error(f"Error in consume: {e}")
        return error_response(str(e), 500)

@app.route('/api/delete/<serial_number>', methods=['DELETE'])
@api_login_required
@admin_required
def delete_meter(serial_number):
    """Delete a meter (admin only)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM meter_data WHERE serial_number = ?", (serial_number,))
        result = cursor.fetchone()

        if result is None:
            conn.close()
            return error_response("Meter not found.", 404)

        cursor.execute("DELETE FROM meter_data WHERE serial_number = ?", (serial_number,))
        conn.commit()
        conn.close()
        
        logger.info(f"Meter deleted: {serial_number}")
        return success_response(message="Meter deleted successfully!")

    except Exception as e:
        logger.error(f"Error deleting meter: {e}")
        return error_response(f"Database error: {e}", 500)

# ==================== PREDICTION ENDPOINTS ====================
@app.route('/predict', methods=['POST'])
@api_login_required
def predict():
    """Make payment prediction based on input data"""
    try:
        if not request.is_json:
            return error_response("Content-Type must be application/json", 400)
            
        data = request.get_json()
        
        prev_payment = data.get("previous_month_payment")
        two_months_payment = data.get("two_months_ago_payment")
        
        if prev_payment is None or two_months_payment is None:
            return error_response("Missing required fields. Please provide previous_month_payment and two_months_ago_payment", 400)
        
        # Use the unified prediction function
        predicted_payment = predict_with_fallback(prev_payment, two_months_payment)
        
        return success_response({
            "predicted_payment": round(float(predicted_payment), 2),
            "method": "ml_model" if loaded_model else "fallback_weighted_average"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return error_response(str(e), 500)

def get_recharge_data(serial_number):
    """Fetch recharge data from database for prediction"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT recharge_amount FROM recharges 
            WHERE serial_number = ? AND recharge_amount > 0
            ORDER BY timestamp DESC LIMIT 2
        ''', (serial_number,))
        recharges = cursor.fetchall()
        conn.close()
        
        return [r['recharge_amount'] for r in recharges]
    except Exception as e:
        logger.error(f"Error getting recharge data: {e}")
        return []

@app.route('/predictx', methods=['POST'])
@api_login_required
def predictx():
    """Make prediction based on meter's recharge history"""
    try:
        if not request.is_json:
            return error_response("Content-Type must be application/json", 400)
            
        data = request.get_json()
        serial_number = data.get('serial_number', '').strip()

        if not serial_number:
            return error_response("Serial number is required.", 400)

        recharge_data = get_recharge_data(serial_number)
        
        if len(recharge_data) < 2:
            return error_response("Insufficient recharge history for prediction. Need at least 2 months of data.", 400)

        # Use the two most recent recharges for prediction
        predicted_payment = predict_with_fallback(recharge_data[0], recharge_data[1])
        
        return success_response({
            "predicted_payment": round(float(predicted_payment), 2),
            "method": "ml_model" if loaded_model else "fallback_weighted_average",
            "data_points": len(recharge_data)
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return error_response(str(e), 500)

# ==================== USER MANAGEMENT API ====================
@app.route('/api/users', methods=['GET'])
@api_login_required
@admin_required
def get_users():
    """Get all users (admin only)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT u.id, u.username, u.email, u.role, u.created_at,
                   m.serial_number as meter_serial
            FROM users u
            LEFT JOIN meter_data m ON u.username = m.owner_name
            ORDER BY u.created_at DESC
        ''')
        users = cursor.fetchall()
        conn.close()
        
        return success_response([dict(user) for user in users])
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        return error_response(str(e), 500)

@app.route('/api/users', methods=['POST'])
@api_login_required
@admin_required
def create_user():
    """Create a new user (admin only)"""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role', 'client')
        meter_serial = data.get('meter_serial')
        
        if not all([username, email, password]):
            return error_response("Missing required fields", 400)
        
        password_hash = generate_password_hash(password)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            conn.close()
            return error_response("Username or email already exists", 400)
        
        cursor.execute('''
            INSERT INTO users (username, password_hash, email, role, created_at)
            VALUES (?, ?, ?, ?, datetime("now"))
        ''', (username, password_hash, email, role))
        
        user_id = cursor.lastrowid
        
        if meter_serial:
            cursor.execute('''
                UPDATE meter_data SET owner_name = ? WHERE serial_number = ?
            ''', (username, meter_serial))
        
        conn.commit()
        conn.close()
        
        logger.info(f"User created: {username} (role: {role})")
        return success_response({"id": user_id}, "User created successfully")
        
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return error_response(str(e), 500)

@app.route('/api/users/<int:user_id>', methods=['PUT'])
@api_login_required
@admin_required
def update_user(user_id):
    """Update a user (admin only)"""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        role = data.get('role')
        meter_serial = data.get('meter_serial')
        new_password = data.get('new_password')
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if username:
            updates.append("username = ?")
            params.append(username)
        if email:
            updates.append("email = ?")
            params.append(email)
        if role:
            updates.append("role = ?")
            params.append(role)
        if new_password:
            updates.append("password_hash = ?")
            params.append(generate_password_hash(new_password))
        
        if updates:
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
            params.append(user_id)
            cursor.execute(query, params)
        
        if meter_serial is not None:
            cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            if user:
                cursor.execute('UPDATE meter_data SET owner_name = NULL WHERE owner_name = ?', (user['username'],))
                if meter_serial:
                    cursor.execute('UPDATE meter_data SET owner_name = ? WHERE serial_number = ?', 
                                 (user['username'], meter_serial))
        
        conn.commit()
        conn.close()
        
        return success_response(message="User updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        return error_response(str(e), 500)

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@api_login_required
@admin_required
def delete_user(user_id):
    """Delete a user (admin only)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if user_id == session['user_id']:
            conn.close()
            return error_response("Cannot delete your own account", 400)
        
        cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if user:
            cursor.execute('UPDATE meter_data SET owner_name = NULL WHERE owner_name = ?', (user['username'],))
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"User deleted: ID {user_id}")
        return success_response(message="User deleted successfully")
        
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return error_response(str(e), 500)

# ==================== USSD ENDPOINTS ====================
@app.route('/ussd', methods=['POST'])
def ussd_callback():
    """Handle USSD requests"""
    session_id = request.form.get("sessionId")
    service_code = request.form.get("serviceCode")
    phone_number = request.form.get("phoneNumber")
    text = request.form.get("text", "")

    user_input = text.strip().split("*") if text else []

    if not text:
        response = "CON Welcome to Energy Meter Service\n"
        response += "1. Check Balance\n"
        response += "2. Recharge Meter\n"
        response += "3. Get Payment Prediction\n"
        response += "4. View Last Recharge"
        return response

    elif text == "1":
        response = "CON Enter meter serial number:"
        return response

    elif len(user_input) == 2 and user_input[0] == "1":
        serial_number = user_input[1]
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT balance FROM meter_data WHERE serial_number = ?", (serial_number,))
        result = cursor.fetchone()
        conn.close()

        if result:
            balance = result['balance']
            response = f"END Your meter balance is: {balance:.2f} RWF"
        else:
            response = "END Meter not found. Please check your serial number."
        return response

    elif text == "2":
        response = "CON Enter meter serial number:"
        return response

    elif len(user_input) == 2 and user_input[0] == "2":
        response = "CON Enter the recharge amount in RWF:"
        return response

    elif len(user_input) == 3 and user_input[0] == "2":
        serial_number = user_input[1]
        recharge_amount = user_input[2]
        
        try:
            recharge_amount = float(recharge_amount)
            credit_amount = recharge_amount * Config.CONVERSION_RATE
            
            if credit_amount <= 0:
                return "END Invalid recharge amount."

            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT balance FROM meter_data WHERE serial_number = ?", (serial_number,))
            row = cursor.fetchone()

            if row:
                current_balance = row['balance']
                new_balance = current_balance + credit_amount

                cursor.execute("UPDATE meter_data SET balance = ? WHERE serial_number = ?", 
                             (new_balance, serial_number))

                cursor.execute('''
                    INSERT INTO recharges (serial_number, recharge_amount, timestamp)
                    VALUES (?, ?, datetime("now"))
                ''', (serial_number, credit_amount))

                conn.commit()
                conn.close()

                return f"END Recharge successful! New balance: {new_balance:.2f} RWF"
            else:
                conn.close()
                return "END Meter not found. Please check your serial number."

        except ValueError:
            return "END Invalid amount. Please enter a valid number."
        except Exception as e:
            logger.error(f"USSD recharge error: {e}")
            return "END An error occurred. Please try again."

    elif text == "3":
        response = "CON Enter meter serial number for prediction:"
        return response

    elif len(user_input) == 2 and user_input[0] == "3":
        serial_number = user_input[1]

        try:
            data = {"serial_number": serial_number}
            
            with app.test_client() as client:
                response = client.post('/predictx', json=data)
                prediction_data = response.get_json()

            if prediction_data and prediction_data.get('success'):
                predicted_payment = prediction_data['data']['predicted_payment']
                method = prediction_data['data'].get('method', 'unknown')
                response = f"END Your predicted next payment is: {predicted_payment} RWF\nMethod: {method}"
            else:
                error_msg = prediction_data.get('error', 'Unknown error') if prediction_data else 'Service unavailable'
                response = f"END Error: {error_msg}"
                
        except Exception as e:
            logger.error(f"USSD prediction error: {e}")
            response = "END Prediction service unavailable"

        return response

    elif text == "4":
        response = "CON Enter meter serial number:"
        return response

    elif len(user_input) == 2 and user_input[0] == "4":
        serial_number = user_input[1]
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT recharge_amount, timestamp FROM recharges 
            WHERE serial_number = ? AND recharge_amount > 0
            ORDER BY timestamp DESC LIMIT 1
        ''', (serial_number,))
        result = cursor.fetchone()
        conn.close()

        if result:
            amount = result['recharge_amount']
            timestamp = result['timestamp']
            response = f"END Last recharge: {amount:.2f} RWF on {timestamp}"
        else:
            response = "END No recharge history found for this meter"
        
        return response

    else:
        response = "END Invalid selection. Please try again."
        return response

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'Endpoint not found'
        }), 404
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500
    return render_template('500.html'), 500

# ==================== HEALTH CHECK ====================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': loaded_model is not None,
        'model_type': type(loaded_model).__name__ if loaded_model else 'None',
        'timestamp': datetime.datetime.now().isoformat()
    })

# ==================== MAIN ====================
if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Get port from environment or use default
    port = int(os.getenv('PORT', 5000))
    
    # Print startup information
    print("\n" + "=" * 60)
    print("ENERGY METER SYSTEM - STARTING UP")
    print("=" * 60)
    print(f"📍 Server running on: http://0.0.0.0:{port}")
    print(f"📁 Database: {Config.DATABASE_PATH}")
    print(f"🤖 ML Model: {'✅ Loaded' if loaded_model else '⚠️ Using Fallback'}")
    if loaded_model:
        print(f"   Model Type: {type(loaded_model).__name__}")
    print("=" * 60 + "\n")
    
    # Run app
    app.run(debug=False, host="0.0.0.0", port=port)
