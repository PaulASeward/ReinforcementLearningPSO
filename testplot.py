import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv("actions_values.csv")  # Replace with your actual file path

# Set style
sns.set_style("whitegrid")

# Plot 1: Distribution of all columns
plt.figure(figsize=(20, 15))
for i, column in enumerate(df.columns):
    plt.subplot(5, 4, i + 1)
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(column)
plt.tight_layout()
plt.show()

# Plot 2: 4 Subplots for groups of 5 columns each
fig, axes = plt.subplots(2, 2, figsize=(20, 15))

for i in range(4):
    row, col = divmod(i, 2)
    subset = df.iloc[:, i * 5:(i + 1) * 5]
    sns.histplot(data=subset, kde=True, bins=30, ax=axes[row, col], element="step")
    axes[row, col].set_title(f"Columns {i*5} to {(i+1)*5 - 1}")

plt.tight_layout()
plt.show()