import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("bioreactor_ml_dataset1.csv")

# Clean column names (remove spaces, unify)
df.columns = df.columns.str.strip()

# Remove incomplete rows
df = df.dropna()

# Derived features
df["Glucose_Consumption_Rate"] = df["Glucose_gL"].diff()
df["DO_Change"] = df["Dissolved_Oxygen_percent"].diff()
df["Specific_Productivity"] = df["Product_Titer_gL"] / df["Cell_Viability_percent"]

# Normalize agitation
scaler = StandardScaler()
df["Agitation_Normalized"] = scaler.fit_transform(df[["Agitation_RPM"]])

# Flags
df["High_Titer_Flag"] = (df["Product_Titer_gL"] > 1).astype(int)
df["Low_Viability_Flag"] = (df["Cell_Viability_percent"] < 98).astype(int)

# Save cleaned dataset
df.to_csv("bioreactor_ml_dataset_cleaned.csv", index=False)

print("✅ Cleaned and enhanced CSV saved as bioreactor_ml_dataset_cleaned.csv")
