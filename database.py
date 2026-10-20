"""SQLite database layer: users, patients, predictions + authentication helpers."""
import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medipredict.db")

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------- security
def hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return salt.hex() + "$" + digest.hex()


def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$")
    check = hash_password(password, bytes.fromhex(salt_hex)).split("$")[1]
    return hmac.compare_digest(check, digest_hex)


def validate_email(email: str):
    return bool(EMAIL_RE.match(email or ""))


def validate_password(password: str):
    """Return list of problems (empty list = valid)."""
    problems = []
    if len(password or "") < 8:
        problems.append("at least 8 characters")
    if not re.search(r"[A-Za-z]", password or ""):
        problems.append("at least one letter")
    if not re.search(r"\d", password or ""):
        problems.append("at least one digit")
    return problems


# ---------------------------------------------------------------- schema
def init_db():
    with get_conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('user','admin')),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                age REAL, pregnancies REAL, glucose REAL, blood_pressure REAL,
                skin_thickness REAL, insulin REAL, bmi REAL, dpf REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                model_name TEXT,
                probability REAL,
                risk_category TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # Default administrator (change this password after first login!)
        row = c.execute("SELECT id FROM users WHERE email='admin@medipredict.local'").fetchone()
        if not row:
            c.execute(
                "INSERT INTO users(email,password_hash,role) VALUES(?,?,'admin')",
                ("admin@medipredict.local", hash_password("Admin@1234")),
            )


# ---------------------------------------------------------------- users
def register_user(email: str, password: str, role: str = "user"):
    email = (email or "").strip().lower()
    if not validate_email(email):
        return False, "Invalid email address."
    problems = validate_password(password)
    if problems:
        return False, "Password needs: " + ", ".join(problems) + "."
    try:
        with get_conn() as c:
            c.execute(
                "INSERT INTO users(email,password_hash,role) VALUES(?,?,?)",
                (email, hash_password(password), role),
            )
        return True, "Account created. You can log in now."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."


def authenticate(email: str, password: str):
    with get_conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", ((email or "").strip().lower(),)).fetchone()
    if row and verify_password(password, row["password_hash"]):
        return dict(row)
    return None


def list_users():
    with get_conn() as c:
        return [dict(r) for r in c.execute("SELECT id,email,role,created_at FROM users ORDER BY id")]


def delete_user(user_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM users WHERE id=?", (user_id,))


# ---------------------------------------------------------------- records
FEATURE_COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "dpf", "age"]


def save_record(user_id, name, values: dict, model_name, probability, risk):
    with get_conn() as c:
        cur = c.execute(
            """INSERT INTO patients(user_id,name,age,pregnancies,glucose,blood_pressure,
               skin_thickness,insulin,bmi,dpf) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (user_id, name, values["age"], values["pregnancies"], values["glucose"],
             values["blood_pressure"], values["skin_thickness"], values["insulin"],
             values["bmi"], values["dpf"]),
        )
        pid = cur.lastrowid
        c.execute(
            "INSERT INTO predictions(patient_id,model_name,probability,risk_category) VALUES(?,?,?,?)",
            (pid, model_name, probability, risk),
        )
    return pid


def update_record(patient_id, name, values: dict, model_name, probability, risk):
    with get_conn() as c:
        c.execute(
            """UPDATE patients SET name=?,age=?,pregnancies=?,glucose=?,blood_pressure=?,
               skin_thickness=?,insulin=?,bmi=?,dpf=? WHERE id=?""",
            (name, values["age"], values["pregnancies"], values["glucose"], values["blood_pressure"],
             values["skin_thickness"], values["insulin"], values["bmi"], values["dpf"], patient_id),
        )
        c.execute("DELETE FROM predictions WHERE patient_id=?", (patient_id,))
        c.execute(
            "INSERT INTO predictions(patient_id,model_name,probability,risk_category) VALUES(?,?,?,?)",
            (patient_id, model_name, probability, risk),
        )


def delete_record(patient_id):
    with get_conn() as c:
        c.execute("DELETE FROM patients WHERE id=?", (patient_id,))


def get_history(user_id=None, search=""):
    """Admins pass user_id=None to see everything."""
    sql = """SELECT p.id AS patient_id, p.name, p.age, p.pregnancies, p.glucose, p.blood_pressure,
                    p.skin_thickness, p.insulin, p.bmi, p.dpf, p.user_id, u.email AS owner,
                    pr.model_name, pr.probability, pr.risk_category, pr.created_at
             FROM patients p
             JOIN predictions pr ON pr.patient_id = p.id
             JOIN users u ON u.id = p.user_id
             WHERE 1=1"""
    args = []
    if user_id is not None:
        sql += " AND p.user_id=?"
        args.append(user_id)
    if search:
        sql += " AND p.name LIKE ?"
        args.append(f"%{search}%")
    sql += " ORDER BY pr.created_at DESC, pr.id DESC"
    with get_conn() as c:
        return [dict(r) for r in c.execute(sql, args)]
