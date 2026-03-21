import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import smtplib
import bcrypt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # loads .env file for local development

# ─────────────────────────────────────────────
# Connection helper (cached for performance)
# ─────────────────────────────────────────────
@st.cache_resource
def _get_conn_str() -> str:
    """Read the connection string once and cache it."""
    try:
        return st.secrets["NEON_DATABASE_URL"]
    except Exception:
        conn_str = os.environ.get("NEON_DATABASE_URL", "")
    if not conn_str:
        raise RuntimeError(
            "NEON_DATABASE_URL not set. Add it to Streamlit Secrets or your .env file."
        )
    return conn_str

@st.cache_resource
def _get_pool():
    """Initialize a Threaded Connection Pool (Min 1, Max 20)."""
    conn_str = _get_conn_str()
    return pool.ThreadedConnectionPool(1, 20, conn_str)

@contextmanager
def get_db_connection():
    """Context manager to check out and return connections to the pool safely."""
    db_pool = _get_pool()
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

# ─────────────────────────────────────────────
# Schema initialisation
# ─────────────────────────────────────────────
def init_db():
    with get_db_connection() as conn:
        conn.autocommit = True  # Needed for DDL (CREATE TABLE) in Postgres
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username   TEXT PRIMARY KEY,
            email      TEXT,
            name       TEXT,
            password   TEXT,
            role       TEXT,
            roles      TEXT,
            approved   BOOLEAN DEFAULT FALSE,
            logged_in  BOOLEAN DEFAULT FALSE
        )
        ''')

        # Config table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS config (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
        ''')

        # Predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id         SERIAL PRIMARY KEY,
            username   TEXT NOT NULL,
            timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            inputs     TEXT NOT NULL,
            results    TEXT NOT NULL,
            model_name TEXT,
            FOREIGN KEY (username) REFERENCES users (username)
        )
        ''')

        # Alerts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            username        TEXT PRIMARY KEY,
            email_enabled   BOOLEAN DEFAULT FALSE,
            target_email    TEXT,
            titer_threshold FLOAT DEFAULT 5.0,
            condition       TEXT DEFAULT 'above',
            smtp_server     TEXT,
            smtp_port       INTEGER DEFAULT 587,
            smtp_user       TEXT,
            smtp_pass       TEXT,
            FOREIGN KEY (username) REFERENCES users (username)
        )
        ''')

        # Turn off autocommit for DML (INSERT, UPDATE etc.)
        conn.autocommit = False

        # Default cookie config
        cursor.execute("SELECT COUNT(*) FROM config WHERE key = 'cookie'")
        if cursor.fetchone()[0] == 0:
            default_cookie = {
                'expiry_days': 30,
                'key': 'setup_key_v2_999',
                'name': 'bionexus_auth_v2_999'
            }
            cursor.execute(
                "INSERT INTO config (key, value) VALUES (%s, %s)",
                ("cookie", json.dumps(default_cookie))
            )

        # Default admin user
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        if cursor.fetchone()[0] == 0:
            admin_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
            cursor.execute('''
                INSERT INTO users (username, email, name, password, role, roles, approved)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (username) DO NOTHING
            ''', (
                "admin",
                "admin@example.com",
                "System Admin",
                admin_hash,
                "admin",
                json.dumps(["admin", "user"]),
                True
            ))

        conn.commit()
        cursor.close()


def save_config(cookie_config):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO config (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            ("cookie", json.dumps(cookie_config))
        )
        conn.commit()
        cursor.close()


def get_config():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key = %s", ("cookie",))
        row = cursor.fetchone()
        cursor.close()
        
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
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO users (username, email, name, password, role, roles, approved)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (username) DO UPDATE SET
            email    = EXCLUDED.email,
            name     = EXCLUDED.name,
            password = EXCLUDED.password,
            role     = EXCLUDED.role,
            roles    = EXCLUDED.roles,
            approved = EXCLUDED.approved
        ''', (username, email, name, password, role, json.dumps(roles), approved))
        conn.commit()
        cursor.close()


def update_user_approval(username, approved):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET approved = %s WHERE username = %s", (approved, username))
        conn.commit()
        cursor.close()


def get_authenticator_config():
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Cookie config
        cursor.execute("SELECT value FROM config WHERE key = %s", ("cookie",))
        cookie_row = cursor.fetchone()
        cookie = json.loads(cookie_row['value']) if cookie_row else {}

        # Users
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        cursor.close()

    usernames = {}
    for row in rows:
        user_data = dict(row)
        user_data['roles'] = json.loads(user_data['roles']) if user_data['roles'] else []
        user_data['approved'] = bool(user_data['approved'])
        user_data['logged_in'] = bool(user_data['logged_in'])
        usernames[user_data['username']] = user_data

    # Fail-safe admin
    if 'admin' not in usernames:
        usernames['admin'] = {
            'username': 'admin',
            'email': 'admin@example.com',
            'name': 'System Admin',
            'password': bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode(),
            'roles': ['admin', 'user'],
            'role': 'admin',
            'approved': True,
            'logged_in': False
        }

    # Force fresh cookie
    cookie['name'] = 'bn_auth_v3_77'
    cookie['key'] = 'bn_key_v3_77'

    return {
        'credentials': {'usernames': usernames},
        'cookie': cookie,
        'pre-authorized': {'emails': []}
    }


def list_users():
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT username, email, name, role, approved FROM users")
        users = [dict(row) for row in cursor.fetchall()]
        cursor.close()
    return users


def delete_user(username):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE username = %s", (username,))
        cursor.execute("DELETE FROM alerts WHERE username = %s", (username,))
        cursor.execute("DELETE FROM users WHERE username = %s", (username,))
        conn.commit()
        cursor.close()


def update_user_role(username, role):
    roles = [role]
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET role = %s, roles = %s WHERE username = %s",
            (role, json.dumps(roles), username)
        )
        conn.commit()
        cursor.close()


def save_prediction(username: str, inputs: Dict[str, Any], results: Dict[str, Any], model_name: str = "Bioreactor_v1"):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (username, inputs, results, model_name, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        ''', (
            username,
            json.dumps(inputs),
            json.dumps(results),
            model_name,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        cursor.close()


def get_user_history(username: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            'SELECT id, timestamp, inputs, results, model_name FROM predictions WHERE username = %s ORDER BY timestamp DESC LIMIT %s OFFSET %s',
            (username, limit, offset)
        )
        rows = cursor.fetchall()
        cursor.close()

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


def get_history_count(username: str) -> int:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE username = %s', (username,))
        count = cursor.fetchone()[0]
        cursor.close()
    return count


def delete_history_item(prediction_id: int, username: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM predictions WHERE id = %s AND username = %s', (prediction_id, username))
        conn.commit()
        cursor.close()


# ─────────────────────────────────────────────
# Alert functions
# ─────────────────────────────────────────────

def get_alert_config(username: str) -> Dict[str, Any]:
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM alerts WHERE username = %s', (username,))
        row = cursor.fetchone()
        cursor.close()

    if row:
        return dict(row)
    return {
        "username": username,
        "email_enabled": False,
        "target_email": "",
        "titer_threshold": 5.0,
        "condition": "above",
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "smtp_user": "",
        "smtp_pass": ""
    }


def save_alert_config(username: str, config: Dict[str, Any]):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (
                username, email_enabled, target_email, titer_threshold, condition,
                smtp_server, smtp_port, smtp_user, smtp_pass
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (username) DO UPDATE SET
                email_enabled   = EXCLUDED.email_enabled,
                target_email    = EXCLUDED.target_email,
                titer_threshold = EXCLUDED.titer_threshold,
                condition       = EXCLUDED.condition,
                smtp_server     = EXCLUDED.smtp_server,
                smtp_port       = EXCLUDED.smtp_port,
                smtp_user       = EXCLUDED.smtp_user,
                smtp_pass       = EXCLUDED.smtp_pass
        ''', (
            username,
            bool(config.get('email_enabled')),
            config.get('target_email'),
            config.get('titer_threshold'),
            config.get('condition'),
            config.get('smtp_server'),
            config.get('smtp_port'),
            config.get('smtp_user'),
            config.get('smtp_pass')
        ))
        conn.commit()
        cursor.close()


def send_email_alert(recipient: str, subject: str, body: str, smtp_config: Dict[str, Any]) -> bool:
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
