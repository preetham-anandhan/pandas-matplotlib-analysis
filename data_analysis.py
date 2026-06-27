import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_mock_csv(filename="employee_data.csv"):
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Employee_ID': range(1001, 1001 + n_samples),
        'Department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], size=n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'Age': np.random.randint(22, 60, size=n_samples),
        'Years_Experience': np.random.randint(1, 20, size=n_samples),
        'Salary': np.random.randint(45000, 135000, size=n_samples),
        'Performance_Score': np.random.uniform(1.0, 5.0, size=n_samples).round(1)
    }
    
    df_mock = pd.DataFrame(data)
    # Inject a realistic correlation: Experience increases salary
    df_mock['Salary'] = df_mock['Salary'] + (df_mock['Years_Experience'] * 2500)
    
    df_mock.to_csv(filename, index=False)
    print(f"✓ Mock CSV dataset created successfully as '{filename}'.\n")

# Generate the file
csv_filename = "employee_data.csv"
generate_mock_csv(csv_filename)



# STEP 2: Load Data & Perform Basic Analysis with Pandas

df = pd.read_csv(csv_filename)

print("--- DATASET OVERVIEW ---")
print(df.head(), "\n")

print("--- DATA TYPES & MISSING VALUES ---")
print(df.info(), "\n")

print("--- STATISTICAL DESCRIPTIONS ---")
print(df.describe(), "\n")

# Core Task: Calculate the average of selected columns
avg_salary = df['Salary'].mean()
avg_age = df['Age'].mean()
avg_perf = df['Performance_Score'].mean()

print("--- BASIC DATA ANALYSIS INSIGHTS ---")
print(f"• Average Employee Salary:          ${avg_salary:,.2f}")
print(f"• Average Employee Age:             {avg_age:.1f} years old")
print(f"• Average Employee Performance:     {avg_perf:.2f} out of 5.0")
print("-" * 36, "\n")



# STEP 3: Data Visualization using Matplotlib and Seaborn

# Set a clean visual style for our charts
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 14})

# --- 1. BAR CHART: Average Salary by Department ---
plt.figure(figsize=(8, 5))
dept_salary = df.groupby('Department')['Salary'].mean().sort_values(ascending=False)
dept_salary.plot(kind='bar', color=['#2b5c8f', '#4682b4', '#6baed6', '#9ecae1'], edgecolor='black')

plt.title('Average Salary Across Departments')
plt.xlabel('Department')
plt.ylabel('Average Salary ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 2. SCATTER PLOT: Years of Experience vs. Salary ---
plt.figure(figsize=(8, 5))
scatter = plt.scatter(df['Years_Experience'], df['Salary'], 
                      c=df['Performance_Score'], cmap='viridis', 
                      alpha=0.8, edgecolors='none', s=60)

plt.title('Impact of Experience on Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
cbar = plt.colorbar(scatter)
cbar.set_label('Performance Score (1-5)')
plt.tight_layout()
plt.show()

# --- 3. HEATMAP: Correlation Matrix of Numeric Features ---
plt.figure(figsize=(7, 5))
# Select only numeric variables for correlation mapping
numeric_cols = ['Age', 'Years_Experience', 'Salary', 'Performance_Score']
correlation_matrix = df[numeric_cols].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
            linewidths=0.5, vmin=-1, vmax=1, square=True)

plt.title('Correlation Matrix of Numeric Features')
plt.tight_layout()
plt.show()

# Clean up the generated file afterward if desired
# os.remove(csv_filename)
