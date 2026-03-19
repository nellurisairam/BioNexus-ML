# BioNexus ML: Bioprocess Intelligence Dashboard

BioNexus ML is a professional-grade Streamlit dashboard for predicting and benchmarking bioreactor performance using Machine Learning. It features enterprise-grade authentication, real-time visualization, and a persistent cloud database.

## 🚀 Key Features

- **ML Prediction**: Load pre-trained scikit-learn pipelines to predict `Product_Titer_gL` from bioprocess parameters.
- **Benchmarking**: Compare live data against models to calculate R², MAE, and RMSE.
- **Enterprise Security**: Login, registration, and admin approval workflows via `streamlit-authenticator` v0.4.2.
- **Persistent Cloud DB**: All users, predictions, and alert configs stored in **Neon Postgres** (no data loss on redeployment).
- **Dynamic Alerts**: Configurable email alerts for threshold-based process monitoring.
- **Session Recording**: Integrated screen recorder for documenting analysis sessions.
- **Data Exploration**: Correlation heatmaps, time-series trends, and distribution plots.

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| **Frontend** | Streamlit 1.55 |
| **ML** | Scikit-Learn, Pandas, NumPy, Joblib |
| **Visualization** | Matplotlib, Seaborn |
| **Authentication** | streamlit-authenticator 0.4.2, bcrypt |
| **Database** | Neon Postgres (cloud, persistent) via psycopg2 |
| **Deployment** | Streamlit Cloud + Google Cloud Run (Docker) |
| **Styling** | Custom CSS — Glassmorphism & Neon Aesthetics |

## 📂 Project Structure

```text
BioNexus-ML/
├── assets/              # Images and branding assets
├── data/                # Sample datasets for testing and prediction
├── models/              # Pre-trained model artifacts (.joblib, .json)
├── notebooks/           # Data exploration and model training workflows
├── .streamlit/          # Streamlit configuration (secrets not committed)
├── .github/workflows/   # GitHub Actions CI/CD for Google Cloud Run
├── app_streamlit.py     # Main Streamlit dashboard
├── database_utils.py    # Neon Postgres connection and all DB helpers
├── Dockerfile           # Container image for Google Cloud Run
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## ⚙️ Setup & Installation

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nellurisairam/BioNexus-ML.git
   cd BioNexus-ML
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the root with your Neon connection string:
   ```
   NEON_DATABASE_URL=postgresql://user:password@host/neondb?sslmode=require
   ```

4. **Run the Dashboard**:
   ```bash
   streamlit run app_streamlit.py
   ```

### Streamlit Cloud Deployment

1. Push code to the `main` branch on GitHub.
2. In the Streamlit Cloud dashboard go to **Settings → Secrets** and add:
   ```toml
   NEON_DATABASE_URL = "postgresql://user:password@host/neondb?sslmode=require"
   ```
3. Streamlit Cloud will auto-deploy on every push to `main`.

### Google Cloud Run Deployment

Automated via GitHub Actions on every push. Requires the following GitHub Secrets:
- `GCP_PROJECT_ID`
- `GCP_SA_KEY`

## 👤 User Roles

| Role | Permissions |
|:---|:---|
| **User** | Predictions, data exploration, history, alerts |
| **Admin** | All user permissions + user management console (approve, revoke, delete, change roles) |

Default admin credentials: **username:** `admin` / **password:** `admin123`

> ⚠️ Change the admin password immediately after first login in production.

## 📄 License

This project is licensed under the MIT License.
