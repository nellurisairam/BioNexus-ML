# 📦 Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🎨 Set Plot Style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

# 📥 Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 📊 Basic Overview
print("✅ Dataset shape:", df.shape)
print("\n🔍 Missing values:\n", df.isnull().sum())

# 🧼 Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].median())  # Future-proof fill
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill Embarked with mode

# 🎯 Survival Rate by Gender
survival_by_gender = df.groupby('Sex')['Survived'].mean()
print("\n📈 Survival Rate by Gender:\n", survival_by_gender)

# 📊 Barplot: Survival by Gender
plt.figure()
ax = sns.barplot(x=survival_by_gender.index, y=survival_by_gender.values, palette='pastel')
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate")
plt.xlabel("Gender")
for i, rate in enumerate(survival_by_gender.values):
    ax.text(i, rate + 0.02, f"{rate:.2f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# 📊 Histogram: Age Distribution
plt.figure()
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 📊 Stacked Histogram: Survival by Age
plt.figure()
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, multiple='stack', palette='Set2')
plt.title("Survival Distribution by Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 📊 Barplot: Survival by Passenger Class
plt.figure()
sns.barplot(x='Pclass', y='Survived', data=df, palette='muted')
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# 📊 Barplot: Survival by Embarkation Port
plt.figure()
sns.barplot(x='Embarked', y='Survived', data=df, palette='Set1')
plt.title("Survival Rate by Embarkation Port")
plt.xlabel("Port of Embarkation")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# 🎻 Violin Plot: Age vs Gender vs Survival
plt.figure()
sns.violinplot(x='Sex', y='Age', hue='Survived', data=df, split=True, palette='Set2')
plt.title("Age Distribution by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# 📦 Boxplot: Fare by Class
plt.figure()
sns.boxplot(x='Pclass', y='Fare', data=df, palette='Set3')
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()

# 📊 Countplot: Class vs Gender
plt.figure()
sns.countplot(x='Pclass', hue='Sex', data=df, palette='pastel')
plt.title("Passenger Count by Class and Gender")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 🔥 Heatmap: Missing Data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.tight_layout()
plt.show()

# 🔗 Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# 🔍 Pairplot: Key Features
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']], hue='Survived', palette='husl')
plt.suptitle("Pairplot of Survival vs Key Features", y=1.02)
plt.show()