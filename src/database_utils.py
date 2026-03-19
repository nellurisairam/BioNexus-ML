import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        email TEXT,
        name TEXT,
        password TEXT,
        role TEXT,
        roles TEXT,
        approved BOOLEAN DEFAULT 0,
        logged_in BOOLEAN DEFAULT 0
    )
    ''')
    
    # Config table (for cookie and other global settings)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')

    # Predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        inputs TEXT NOT NULL,
        results TEXT NOT NULL,
        model_name TEXT,
        FOREIGN KEY (username) REFERENCES users (username)
    )
    ''')

    # Alerts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS alerts (
        username TEXT PRIMARY KEY,
        email_enabled BOOLEAN DEFAULT 0,
        target_email TEXT,
        titer_threshold FLOAT DEFAULT 5.0,
        condition TEXT DEFAULT 'above',
        smtp_server TEXT,
        smtp_port INTEGER DEFAULT 587,
        smtp_user TEXT,
        smtp_pass TEXT,
        FOREIGN KEY (username) REFERENCES users (username)
    )
    ''')
    
    # Default configuration if empty
    cursor.execute("SELECT COUNT(*) FROM config WHERE key = 'cookie'")
    if cursor.fetchone()[0] == 0:
        default_cookie = {
            'expiry_days': 30,
            'key': 'setup_key_v2_999',
            'name': 'bionexus_auth_v2_999'
        }
        cursor.execute("INSERT INTO config (key, value) VALUES (?, ?)", 
                       ("cookie", json.dumps(default_cookie)))

    # Default admin user if missing
    cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
    if cursor.fetchone()[0] == 0:
        # Standard bcrypt hash for 'admin123'
        admin_hash = "$2y$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGGa31S."
        cursor.execute('''
            INSERT OR REPLACE INTO users (username, email, name, password, role, roles, approved)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ("admin", "admin@example.com", "System Admin", admin_hash, "admin", json.dumps(["admin", "user"]), 1))

    conn.commit()
    conn.close()

def save_config(cookie_config):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", 
                   ("cookie", json.dumps(cookie_config)))
    conn.commit()
    conn.close()

def get_config():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM config WHERE key = ?", ("cookie",))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return {
        'expiry_days': 1,
        'key': 'some_very_secret_key',
        'name': 'bioreactor_dashboard'
    }

def add_user(username, email, password, name=None, role='user', roles=None, approved=False):
    if roles is None:
        roles = [role]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT OR REPLACE INTO users (username, email, name, password, role, roles, approved)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (username, email, name, password, role, json.dumps(roles), 1 if approved else 0))
    conn.commit()
    conn.close()

def update_user_approval(username, approved):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET approved = ? WHERE username = ?", (1 if approved else 0, username))
    conn.commit()
    conn.close()

def get_authenticator_config():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Load cookie config
    cursor.execute("SELECT value FROM config WHERE key = ?", ("cookie",))
    cookie_row = cursor.fetchone()
    cookie = json.loads(cookie_row['value']) if cookie_row else {}

    # Load users
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()
    
    usernames = {}
    for row in rows:
        user_data: dict[str, Any] = dict(row)
        # Parse logic for roles/role
        user_data['roles'] = json.loads(user_data['roles']) if user_data['roles'] else []
        user_data['approved'] = bool(user_data['approved'])
        user_data['logged_in'] = bool(user_data['logged_in'])
        usernames[user_data['username']] = user_data
    
    # Fail-safe: Ensure admin exists in the returned config
    if 'admin' not in usernames:
        usernames['admin'] = {
            'username': 'admin',
            'email': 'admin@example.com',
            'name': 'System Admin',
            'password': '$2b$12$TRtouVHjBrfeC72JUq.TauKlkNTByD4ZqfQ8bddHwkioFnIJwvwS6',
            'roles': ['admin', 'user'],
            'role': 'admin',
            'approved': True,
            'logged_in': False
        }
    
    # Force new cookie to reset sessions
    cookie['name'] = 'bn_auth_v3_77'
    cookie['key'] = 'bn_key_v3_77'
    
    conn.close()
    
    return {
        'credentials': {'usernames': usernames},
        'cookie': cookie,
        'preauthorized': {'emails': []}
    }

def list_users():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT username, email, name, role, approved FROM users")
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users

def delete_user(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()

def update_user_role(username, role):
    roles = [role]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET role = ?, roles = ? WHERE username = ?", (role, json.dumps(roles), username))
    conn.commit()
    conn.close()

def save_prediction(username: str, inputs: Dict[str, Any], results: Dict[str, Any], model_name: str = "Bioreactor_v1"):
    """Save a prediction result to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (username, inputs, results, model_name, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        username, 
        json.dumps(inputs), 
        json.dumps(results), 
        model_name,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def get_user_history(username: str) -> List[Dict[str, Any]]:
    """Retrieve prediction history for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, inputs, results, model_name FROM predictions WHERE username = ? ORDER BY timestamp DESC', (username,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "id": row['id'],
            "timestamp": row['timestamp'],
            "inputs": json.loads(row['inputs']),
            "results": json.loads(row['results']),
            "model_name": row['model_name']
        })
    return history

def delete_history_item(prediction_id: int, username: str):
    """Delete a specific history item if it belongs to the user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions WHERE id = ? AND username = ?', (prediction_id, username))
    conn.commit()
    conn.close()

# --- Email Alert Functions ---

def get_alert_config(username: str) -> Dict[str, Any]:
    """Retrieve alert configuration for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM alerts WHERE username = ?', (username,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return {
        "username": username,
        "email_enabled": 0,
        "target_email": "",
        "titer_threshold": 5.0,
        "condition": "above",
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "",
        "smtp_pass": ""
    }

def save_alert_config(username: str, config: Dict[str, Any]):
    """Save or update alert configuration for a user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO alerts (
            username, email_enabled, target_email, titer_threshold, condition, 
            smtp_server, smtp_port, smtp_user, smtp_pass
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        username,
        1 if config.get('email_enabled') else 0,
        config.get('target_email'),
        config.get('titer_threshold'),
        config.get('condition'),
        config.get('smtp_server'),
        config.get('smtp_port'),
        config.get('smtp_user'),
        config.get('smtp_pass')
    ))
    conn.commit()
    conn.close()

def send_email_alert(recipient: str, subject: str, body: str, smtp_config: Dict[str, Any]) -> bool:
    """Send an email alert using SMTP."""
    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_config.get('smtp_user')
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(smtp_config.get('smtp_server'), smtp_config.get('smtp_port'))
        server.starttls()
        server.login(smtp_config.get('smtp_user'), smtp_config.get('smtp_pass'))
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False
