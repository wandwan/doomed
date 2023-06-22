import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/camel/VSC/ICR/DataProcessing')
from convToNum import convert_to_numeric
from fillNan import fill_na_with_kmeans
from makeInt import convert_to_integer

# Load the dataframe
df = pd.read_csv('./data/train.csv')

# Create a bar plot of "EJ" and "Class"
df.groupby(['EJ', 'Class']).size().unstack().plot(kind='bar', stacked=True)

# Calculate the ratio between Class 0/1 for each EJ value
df.groupby(['EJ', 'Class']).size().groupby(level=0).apply(lambda x: x / x.sum()).unstack().plot(kind='bar', stacked=True)

# Print the ratio between Class 0/1 for each EJ value
print(df.groupby(['EJ', 'Class']).size().groupby(level=0).apply(lambda x: x / x.sum()).unstack())

df = convert_to_numeric(df)
df = fill_na_with_kmeans(df)
df = convert_to_integer(df)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
df[df['Class'] == 0]['BN'].hist(bins=100, ax=axs[0])
df[df['Class'] == 1]['BN'].hist(bins=100, ax=axs[1])
axs[0].set_title('Class 0')
axs[1].set_title('Class 1')
plt.show()

# Print the unique values of BN
unique_values = sorted(df['BN'].unique())
print(unique_values)

# For all other integer values, print them and their unique values if < 300

# Find all integer columns in df
int_cols = df.select_dtypes(include='integer').columns

# Loop through each integer column
for col in int_cols:
    # Check if the number of unique values is less than 300
    if df[col].nunique() < 300:
        # Print the unique values sorted
        unique_values = sorted(df[col].unique())
        print(f"Unique values of {col}: {unique_values}")

from math import ceil
# Plot the distributions of columns with under 300 unique values
fig, axs = plt.subplots(5, ceil(len(df.select_dtypes(include='integer').columns[df.select_dtypes(include='integer').nunique() < 300]) / 5), figsize=(20, 5))

# Loop through each integer column with less than 300 unique values
for i, col in enumerate(df.select_dtypes(include='integer').columns[df.select_dtypes(include='integer').nunique() < 300]):
    # Plot the histogram of the column
    axs[i//ceil(len(df.select_dtypes(include='integer').columns[df.select_dtypes(include='integer').nunique() < 300]) / 5), i%ceil(len(df.select_dtypes(include='integer').columns[df.select_dtypes(include='integer').nunique() < 300]) / 5)].hist(df[col], bins=100)
    axs[i//ceil(len(df.select_dtypes(include='integer').columns[df.select_dtypes(include='integer').nunique() < 300]) / 5), i%ceil(len(df.select_dtypes(include='integer').columns[df.select_dtypes(include='integer').nunique() < 300]) / 5)].set_title(col, loc='right')


# Show the plot
plt.show()