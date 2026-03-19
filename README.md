# BioNexus ML: Bioprocess Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red?logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/Neon-Postgres-green?logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

**BioNexus ML** is a professional-grade Streamlit application designed for predicting and benchmarking bioreactor performance utilizing Machine Learning. The platform features robust authentication workflows, real-time data visualization, cloud-persistent data storage, and a comprehensive administrative management console.

</div>

---

## End-to-End Architecture

```mermaid
flowchart TD
    %% Styling
    classDef user fill:#ff9900,stroke:#333,stroke-width:2px,color:#fff
    classDef auth fill:#e91e63,stroke:#333,stroke-width:2px,color:#fff
    classDef db fill:#00c853,stroke:#333,stroke-width:2px,color:#fff
    classDef cloud fill:#29b6f6,stroke:#333,stroke-width:2px,color:#000
    classDef action fill:#ab47bc,stroke:#333,stroke-width:2px,color:#fff
    classDef page fill:#455a64,stroke:#333,stroke-width:2px,color:#fff

    A([User / Administrator]):::user --> B{Authenticated?}:::auth

    subgraph Auth ["Authentication Flow"]
        B -- "No" --> C[Login Form]:::auth
        C -- "Invalid Credentials" --> C
        
        B -- "No" --> D[Registration Form]:::auth
        D -- "Submit" --> E[Database Registration]:::db
        E --> F[Pending Administrative Review]:::page
        
        C -- "Valid" --> G{Access Verification}:::db
        G -- "Pending" --> F
        G -- "Approved" --> H[Dashboard Initialized]:::page
    end

    H --> I{Authorization Level?}:::auth

    subgraph Admin ["Administrative Console"]
        I -- "Administrator" --> J[User Management Interface]:::page
        J --> K[Approve/Revoke Access]:::action
        J --> L[Modify User Permissions]:::action
        J --> M[Remove Accounts]:::action
    end

    subgraph App ["Core Dashboard Application"]
        I -- "User/Administrator" --> N[Predictive Analytics]:::page
        I -- "User/Administrator" --> O[Model Training Module]:::page
        I -- "User/Administrator" --> P[Data Visualization]:::page
        I -- "User/Administrator" --> Q[Session History]:::page
        I -- "User/Administrator" --> R[System Alerts]:::page
    end

    subgraph ML ["Machine Learning Pipeline"]
        N -- "Dataset Upload" --> S[Feature Engineering]:::action
        S --> T[Random Forest Assessor]:::action
        T --> U[Product Titer Calculation]:::action
    end

    DB[(Neon PostgreSQL\nCloud Database)]:::db

    %% Data Flow Dependencies
    K -.->|Access Mutator| DB
    L -.->|Permission Mutator| DB
    M -.->|Account Purge| DB
    U -.->|Telemetry Record| DB
    Q <.->|History Query/Delete| DB
    R -.->|Configuration Update| DB
```

---

## Core Capabilities

| Capability | Description |
|:---|:---|
| **Predictive Analytics** | Process uploaded datasets to generate real-time `Product_Titer_gL` estimations utilizing serialized scikit-learn models. |
| **Model Benchmarking** | Quantify predictive accuracy against live process data via statistical metrics (R², MAE, RMSE). |
| **In-Application Training** | Facilitate the training and optimization of Random Forest Regressor models directly from the user interface. |
| **Data Visualization** | Render complex multivariate relationships via correlation matrices, time-series analysis, and kernel density estimations. |
| **Audit Logging** | Maintain and provide user-specific access to historical prediction telemetry and model interactions. |
| **Documentation** | Integrated reference manual detailing critical process parameters and predictive confidence intervals. |
| **Event Thresholding** | Configure automated SMTP-based notifications triggered by predefined biometric threshold deviations. |
| **Session Documentation** | Native screen recording functionality to archive exploratory data analysis sessions. |
| **Security & Access** | Role-Based Access Control (RBAC), secure registration, and administrative verification managed via `streamlit-authenticator` v0.4.2. |
| **Cloud Persistence** | Centralized, durable storage for user credentials, operational telemetry, and configuration preferences using Neon PostgreSQL. |

---

## Technology Stack

| Component | Technology | Version |
|:---|:---|:---|
| **Frontend Framework** | Streamlit | 1.55.0 |
| **Machine Learning** | Scikit-Learn | 1.7.2 |
| **Data Aggregation** | Pandas, NumPy | 2.3.3 / 2.2.6 |
| **Graphical Rendering** | Matplotlib, Seaborn | 3.10.6 / 0.13.2 |
| **Binary Serialization** | Joblib | 1.5.2 |
| **Authentication Module** | streamlit-authenticator | 0.4.2 |
| **Cryptographic Hashing** | bcrypt | 5.0.0 |
| **Database Engine** | Neon PostgreSQL via psycopg2 | 2.9.10 |
| **Environment Configuration** | python-dotenv | 1.2.2 |
| **Application Hosting** | Streamlit Cloud | — |

---

## Project Architecture

```text
BioNexus-ML/
├── assets/                  # Graphical assets and branding collateral
├── data/                    # Benchmark datasets for validation
├── models/                  # Serialized ML pipelines (.joblib) and schema definitions (.json)
├── notebooks/               # Experimental Jupyter environments
├── .streamlit/
│   └── secrets.toml         # Environment credentials (excluded from version control)
├── app_streamlit.py          # Primary application entry point
├── database_utils.py         # PostgreSQL connection protocols and database operations
├── requirements.txt          # Defined Python dependencies
├── start_app.bat             # Windows execution script
├── .env                      # Local environment variables (excluded from version control)
├── .gitignore                # Version control exclusions
└── README.md                 # System documentation
```

---

## Application Workspaces

### 1. Predict & Benchmark
- Ingest bioreactor run data via CSV upload.
- Execute automated data preprocessing and feature engineering (e.g., consumption rate derivation, variance normalization).
- Compute outcome estimations for `Product_Titer_gL` via the active predictive pipeline.
- Render statistical accuracy visualizations encompassing R², MAE, RMSE, and associated Feature Importance graphs.

### 2. Train Model
- Upload verified historical datasets for localized model training.
- Automatically construct and evaluate a Random Forest Regressor algorithm.
- Provide post-training statistical performance summaries.
- Serialize the derived model for subsequent application use.

### 3. Data Exploration
- Perform descriptive statistical analysis across all process parameters.
- Visualize parameter correlation via annotated heatmaps.
- Track longitudinal process variations utilizing multi-axis time-series plotting.
- Examine continuous data distribution via histogram and generic box plots.

### 4. Prediction History
- Display a chronological audit trail of all predictive computations executed by the authenticated user.
- Detail the associated predictive model and core input parameters per session.
- Manage compliance by allowing session deletion.

### 5. Interpretation Guide
- Access the platform's standard operating procedure (SOP) documentation.
- Review operational definitions for critical identifiers (Temperature, pH, DO%, Agitation).
- Understand the confidence boundaries associated with the active predictive models.

### 6. System Alerts
- Establish rule-based conditionals for asynchronous status updates via email.
- Set thresholds (`above`/`below`) tied to primary metrics globally.
- Persist notification profiles directly to the remote PostgreSQL instance.

---

## Administrative Operations

The platform enforces Role-Based Access Control distinguishing standard Users from Administrators.

### Administrative Console
Administrators possess exclusive access to the sidebar management interface, facilitating:
- **Pending Approvals**: Reviewing and authorizing newly registered accounts.
- **Access Control**: Promoting user permissions to administrative levels or revoking access entirely.
- **Account Deletion**: Managing identity lifecycles globally.

> **Note on Initial Deployment:**
> The default administrative account is provisioned as follows:
> - Username: `admin`
> - Password: `admin123`
>
> Ensure these credentials are updated immediately upon deploying to a production environment.

---

## System Initialization & Deployment

### Prerequisites
- Python 3.10 or higher
- An active Neon PostgreSQL database instance

### Local Environment Setup

1. **Repository Access**:
   ```bash
   git clone https://github.com/nellurisairam/BioNexus-ML.git
   cd BioNexus-ML
   ```

2. **Environment Isolation**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

3. **Dependency Resolution**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration Mapping**:
   Create a `.env` file in the project root containing the respective Data Source Name (DSN) string:
   ```env
   NEON_DATABASE_URL=postgresql://user:password@host/neondb?sslmode=require
   ```

5. **Application Execution**:
   ```bash
   streamlit run app_streamlit.py
   ```
   (Alternatively via Windows: execute `start_app.bat`.)

---

### Streamlit Cloud Deployment

1. Push the committed repository to the `main` branch.
2. Link the repository via the Streamlit Cloud administrative portal.
3. Configure the **Application Entrypoint** to `app_streamlit.py`.
4. Ensure the Python runtime environment is set to `3.10`.
5. Under **Advanced Settings → Secrets**, inject the database environment variable:
   ```toml
   NEON_DATABASE_URL = "postgresql://user:password@host/neondb?sslmode=require"
   ```
6. The platform will automatically trigger a redeployment upon subsequent commits to `main`.

---

## Data Architecture

| Persistent Table | Functional Description |
|:---|:---|
| `users` | Maintains identity profiles inclusive of hashed credentials and RBAC assignments |
| `config` | Provisions persistent application-wide configuration arrays (e.g., cookie parameters) |
| `predictions` | Logs transactional histories regarding user-driven model estimations |
| `alerts` | Stores threshold configurations and target delivery definitions for asynchronous alerts |

---

## Security Directives

- **Credential Exclusions**: Configuration files detailing database Uniform Resource Identifiers (URIs) are excluded from version control parsing.
- **Cryptographic Standards**: Authentication passwords are mathematically hashed locally using `bcrypt` implementation combined with dynamic salting prior to database transmission.
- **Session Management**: JWT or cookie-based state persistence is secured entirely through the validated methods provided by `streamlit-authenticator`.

---

## License

This architecture is released and distributed under the MIT License — refer to the `LICENSE` document for legal delineations.
