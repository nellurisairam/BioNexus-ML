# 🔬 BioNexus ML: Bioprocess Intelligence Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/Neon-Postgres-green?logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

**BioNexus ML** is a professional-grade Streamlit dashboard for predicting and benchmarking bioreactor performance using Machine Learning. It features enterprise-grade authentication, real-time data visualization, cloud-persistent data, and a full administrative console — deployed on Streamlit Cloud with Neon Postgres as the backend database.

</div>

---

## 🗺️ End-to-End Architecture Flowchart

```mermaid
flowchart TD
    %% Styling
    classDef user fill:#ff9900,stroke:#333,stroke-width:2px,color:#fff
    classDef auth fill:#e91e63,stroke:#333,stroke-width:2px,color:#fff
    classDef db fill:#00c853,stroke:#333,stroke-width:2px,color:#fff
    classDef cloud fill:#29b6f6,stroke:#333,stroke-width:2px,color:#000
    classDef action fill:#ab47bc,stroke:#333,stroke-width:2px,color:#fff
    classDef page fill:#455a64,stroke:#333,stroke-width:2px,color:#fff

    A([👤 User / Admin]):::user --> B[Open App URL\nStreamlit Cloud / localhost]:::cloud
    B --> C{Authenticated?}:::auth

    C -- No --> D[Login Form\nSidebar]:::auth
    D -- Invalid --> D
    D -- Valid --> E[Check Approval Status\nNeon Postgres]:::db
    E -- Pending --> F[⏳ Awaiting Admin Approval]:::auth
    E -- Approved --> G[🏠 Dashboard Loaded]:::page

    C -- No --> H[Register New Account\nSidebar Form]:::auth
    H --> I[Save to Neon Postgres\napproved = false]:::db
    I --> F

    G --> J{User Role?}:::auth

    J -- Admin --> K[🔑 Admin Console\nSidebar]:::page
    K --> K1[Approve / Revoke Users]:::action
    K --> K2[Change Roles]:::action
    K --> K3[Delete Users]:::action

    J -- All Users --> L[Dashboard Tabs]:::page

    L --> T1[🚀 Predict & Benchmark\nUpload CSV → ML Pipeline\n→ Titer Prediction + Metrics]:::page
    L --> T2[🏋️ Train Model\nUpload Training CSV\n→ Fit Random Forest]:::page
    L --> T3[📊 Data Exploration\nCorrelation Heatmap\nTime-Series / Distributions]:::page
    L --> T4[📜 Prediction History\nView & Delete Past Sessions]:::page
    L --> T5[📘 Interpretation Guide\nProcess Parameter Reference]:::page
    L --> T6[🔔 Alerts\nConfigure Email SMTP\nThreshold Triggers]:::page

    T1 --> DB[(🗄️ Neon Postgres\ncloud persistent)]:::db
    T4 --> DB
    T6 --> DB
    K1 --> DB
    K2 --> DB
    K3 --> DB

    subgraph Deployment["☁️ Deployment"]
        SC[Streamlit Cloud\nauto-deploy on git push to main]:::cloud
        DB
    end

    subgraph ML Pipeline
        M1[Pre-trained .joblib Model]:::action
        M2[Feature Engineering\nGlucose Rate, DO Change,\nSpecific Productivity, Flags]:::action
        M3[Prediction Output\nProduct Titer g/L]:::action
        M2 --> M1 --> M3
    end

    T1 --> M2
```

---

| Feature | Details |
|:---|:---|
| **ML Prediction** | Upload CSV data and get instant `Product_Titer_gL` predictions from trained scikit-learn pipelines |
| **Model Benchmarking** | Compare live data against the model to compute R², MAE, RMSE |
| **Model Training** | Train a new Random Forest model directly from the UI |
| **Data Exploration** | Correlation heatmaps, time-series trends, distribution plots, and summary statistics |
| **Prediction History** | View and delete past prediction sessions per user |
| **Interpretation Guide** | Built-in documentation for understanding process parameters and model outputs |
| **Dynamic Alerts** | Configure email alerts (SMTP) triggered when `Product_Titer_gL` crosses a threshold |
| **Session Recording** | Built-in screen recorder that downloads a `.webm` video of your analysis session |
| **Enterprise Auth** | Login, Registration, Admin Approval via `streamlit-authenticator` v0.4.2 |
| **Persistent Database** | All users, predictions, and alert configs stored in **Neon Postgres** (persists across redeploys) |
| **Responsive Design** | Glassmorphism & neon UI, fully responsive for mobile, tablet, and desktop |
| **Theme Toggle** | Light and Dark mode switcher in the sidebar |

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|:---|:---|:---|
| **Frontend** | Streamlit | 1.55.0 |
| **Machine Learning** | Scikit-Learn | 1.7.2 |
| **Data Processing** | Pandas, NumPy | 2.3.3 / 2.2.6 |
| **Visualization** | Matplotlib, Seaborn | 3.10.6 / 0.13.2 |
| **Model Serialization** | Joblib | 1.5.2 |
| **Authentication** | streamlit-authenticator | 0.4.2 |
| **Password Hashing** | bcrypt | 5.0.0 |
| **Database** | Neon Postgres (cloud) via psycopg2 | 2.9.10 |
| **Env Management** | python-dotenv | 1.2.2 |
| **Deployment** | Streamlit Cloud | — |
| **Styling** | Custom CSS — Glassmorphism & Neon Aesthetics | — |

---

## 📂 Project Structure

```text
BioNexus-ML/
├── assets/                  # Background image and branding assets
├── data/                    # Sample CSV datasets for testing and prediction
├── models/                  # Pre-trained model artifacts (.joblib, .json schema)
├── notebooks/               # Jupyter notebooks for data exploration & model training
├── .streamlit/
│   └── secrets.toml         # 🔒 Local secrets (NOT committed to Git)
├── app_streamlit.py          # Main Streamlit application (all UI and logic)
├── database_utils.py         # Neon Postgres connection & all DB operations
├── requirements.txt          # Python dependencies
├── start_app.bat             # Windows batch file for local launch
├── .env                      # 🔒 Local env vars (NOT committed to Git)
├── .gitignore                # Excludes .env, *.db, secrets.toml, __pycache__
└── README.md                 # Project documentation
```

---

## 📊 Dashboard Tabs

### 🚀 Tab 1 — Predict & Benchmark
- Upload a CSV file with bioreactor run data.
- The app automatically preprocesses the data (derived features, normalization, flags).
- Generates predictions for `Product_Titer_gL` using the loaded ML pipeline.
- Displays benchmark metrics: **R²**, **MAE**, **RMSE**, **Max Error**.
- Shows a **Feature Importance** chart for tree-based models.
- Saves each prediction session to the database (Neon Postgres).

### 🏋️ Tab 2 — Train Model
- Upload a labeled training CSV directly from the UI.
- Trains a **Random Forest Regressor** pipeline with automatic feature engineering.
- Evaluates the model and shows training stats.
- Saves the new model artifact to the `models/` directory.

### 📊 Tab 3 — Data Exploration
- Upload or use sample data to explore your dataset.
- Summary statistics (min, max, mean, std) per column.
- **Correlation Heatmap** (Seaborn, annotated).
- **Time-Series Trends** — multi-parameter line chart over `Time_hours`.
- **Distribution Plots** — histogram + KDE and box plot per feature.
- Metric summary cards: Avg Temperature, pH, Dissolved Oxygen, Product Titer.

### 📜 Tab 4 — History
- Lists all prediction sessions made by the logged-in user.
- Shows timestamp, model used, and key inputs.
- Option to delete individual history entries.

### 📘 Tab 5 — Interpretation Guide
- Built-in reference guide explaining:
  - What each input parameter means (Temperature, pH, DO%, RPM, etc.)
  - How to interpret model outputs (Titer prediction range, confidence).
  - Best practices for bioreactor process interpretation.

### 🔔 Tab 6 — Alerts
- Configure email notifications when `Product_Titer_gL` crosses a set threshold.
- Supports **above** or **below** conditions.
- SMTP configuration (works with Gmail, Outlook, etc.).
- Settings saved per user in Neon Postgres.

---

## 👤 User Roles & Admin Console

| Role | Capabilities |
|:---|:---|
| **User** | Predictions, benchmarking, training, data exploration, history, alerts, screen recording |
| **Admin** | All user capabilities + Admin Management Console |

### 🔑 Admin Management Console (Sidebar)
- **Pending Approvals**: See and approve new user registrations.
- **User Management**: View all users with email and role, change roles (user/admin), approve/revoke access, delete users.

> **Default Admin Credentials:**
> - Username: `admin`
> - Password: `admin123`
>
> ⚠️ Change the admin password immediately after first login in production!

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- A [Neon](https://neon.tech) Postgres database (free tier is sufficient)

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nellurisairam/BioNexus-ML.git
   cd BioNexus-ML
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** in the root with your Neon connection string:
   ```env
   NEON_DATABASE_URL=postgresql://user:password@host/neondb?sslmode=require
   ```

5. **Run the Dashboard**:
   ```bash
   streamlit run app_streamlit.py
   ```
   Or on Windows, double-click `start_app.bat`.

---

## ☁️ Cloud Deployment

### Streamlit Cloud

1. Push your code to the `main` branch on GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set the **main file path** to `app_streamlit.py`.
4. Set the **Python version** to `3.10` in the app settings.
5. Go to **Settings → Secrets** and add:
   ```toml
   NEON_DATABASE_URL = "postgresql://user:password@host/neondb?sslmode=require"
   ```
246: 6. Streamlit Cloud auto-redeploys on every push to `main`.
247: 
248: ---

## 🔒 Security Notes

- **Secrets**: The `.env` and `.streamlit/secrets.toml` files are excluded from Git via `.gitignore`.
- **Password Hashing**: All passwords are hashed using `bcrypt` with a unique salt per user.
- **Admin Approval**: New registrations require admin approval before they can access the dashboard.
- **Session Cookies**: Authentication cookies are managed securely by `streamlit-authenticator`.

---

## 🗄️ Database Schema (Neon Postgres)

| Table | Description |
|:---|:---|
| `users` | Stores username, hashed password, email, role, approval status |
| `config` | Stores cookie configuration |
| `predictions` | Stores per-user prediction history (inputs, results, timestamp) |
| `alerts` | Stores per-user email alert configuration |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Built with ❤️ for Bioprocess Engineers
</div>
