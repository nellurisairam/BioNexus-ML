# BioNexus ML: Bioprocess Intelligence Dashboard

BioNexus ML is a professional-grade Streamlit dashboard designed for predicting and benchmarking bioreactor performance using Machine Learning. It features a robust authentication system, real-time data visualization, and an administrative console for user management.

![Dashboard Preview](assets/background.png)

## 🚀 Key Features

- **ML Prediction**: Load pre-trained scikit-learn pipelines to predict `Product_Titer_gL` from process parameters.
- **Benchmarking**: Compare live process data against models to calculate R², MAE, and RMSE.
- **Enterprise Security**: Secure login and registration with admin approval workflows.
- **Intelligence Guide**: Built-in guide for interpreting process parameters and model outputs.
- **Session Recording**: Integrated screen recorder for documenting analysis sessions.
- **Dynamic Alerts**: Configurable email alerts for threshold-based process monitoring.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **Authentication**: Streamlit-Authenticator, YAML, SQLite
- **Styling**: Custom CSS (Glassmorphism & Neon Aesthetics)

## 📂 Project Structure

```text
BioNexus-ML/
├── assets/             # Images, icons, and branding assets
├── data/               # Sample datasets for testing and prediction
├── models/             # Pre-trained model artifacts (.joblib, .json)
├── notebooks/          # Data exploration and model training workflows
├── app_streamlit.py    # Main Streamlit dashboard
├── database_utils.py  # Database and authentication helpers
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/BioNexus-ML.git
   cd BioNexus-ML
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run app_streamlit.py
   ```

## 👤 User Roles

- **User**: Can perform predictions, view history, and explore data.
- **Admin**: Full access + User management console for approving new registrations and managing roles.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
