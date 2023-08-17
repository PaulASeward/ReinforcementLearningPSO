import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns


# Read the CSV files
global_df = pd.read_csv('GlobalVectorizedfunctionresults.csv')
local_df = pd.read_csv('LocalVectorizedfunctionresults.csv')
iterative_df = pd.read_csv('LocalIterativefunctionresults.csv')

# Remove outlier from Function Result 2
global_df = global_df[global_df['Function Number'] != 2]
local_df = local_df[local_df['Function Number'] != 2]
iterative_df = iterative_df[iterative_df['Function Number'] != 2]

# Remove outlier from Function Result 3
global_df = global_df[global_df['Function Number'] != 3]
local_df = local_df[local_df['Function Number'] != 3]
iterative_df = iterative_df[iterative_df['Function Number'] != 3]

# Create comparison tables
averageds_df = pd.DataFrame({
    'Approach': ['Global Vectorized', 'Local Vectorized', 'Local Iterative'],
    'Mean Error': [
        global_df['Mean Error'].mean(),
        local_df['Mean Error'].mean(),
        iterative_df['Mean Error'].mean()
    ],
    'Standard Deviation': [
        global_df['Standard Deviation'].mean(),
        local_df['Standard Deviation'].mean(),
        iterative_df['Standard Deviation'].mean()
    ],
    'Mean Time': [
        global_df['Mean Time'].mean(),
        local_df['Mean Time'].mean(),
        iterative_df['Mean Time'].mean()
    ]
})

# Format the output tables
mean_error_table = tabulate(averageds_df[['Approach', 'Mean Error']], headers='keys', tablefmt='pipe', showindex=False)
std_dev_table = tabulate(averageds_df[['Approach', 'Standard Deviation']], headers='keys', tablefmt='pipe', showindex=False)
mean_time_table = tabulate(averageds_df[['Approach', 'Mean Time']], headers='keys', tablefmt='pipe', showindex=False)

# Print the formatted output tables
print('Mean Error Comparison:')
print(mean_error_table)
print()

print('Standard Deviation Comparison:')
print(std_dev_table)
print()

print('Mean Time Comparison:')
print(mean_time_table)


# Remove the first column from local_df and iterative_df
local_df = local_df.iloc[:, 1:]
iterative_df = iterative_df.iloc[:, 1:]


# Combine the dataframes
combined_df = pd.concat([global_df, local_df, iterative_df], axis=1, keys=['Global Vectorized', 'Local Vectorized', 'Local Iterative'])

# Rename the columns to make them more readable
combined_df.columns = ['Function Number', 'Mean Error (Global)', 'Std Deviation (Global)', 'Mean Time (Global)',
                       'Mean Error (Local)', 'Std Deviation (Local)', 'Mean Time (Local)',
                       'Mean Error (Iterative)', 'Std Deviation (Iterative)', 'Mean Time (Iterative)']

# Generate the table
table = tabulate(combined_df, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='center')

# Print the table
print()
print(table)

# # Save the table to an Excel file
# combined_df.to_csv('Overall_function_results.csv')

# # Convert the combined dataframe to a list of lists
# table_data = combined_df.values.tolist()
#
# # Get the column headers
# headers = combined_df.columns.levels[1].tolist()
#
#
# # Generate the table
# table = tabulate(table_data, headers, tablefmt='fancy_grid', numalign='center')

# # Print the table
# print(table)




## BOXPLOT
# # Use seaborn for better aesthetics (optional)
# # import seaborn as sns
# # sns.set(style="whitegrid")
#
# # Add an "Approach" column to each DataFrame
# global_df['Approach'] = 'Global Vectorized'
# local_df['Approach'] = 'Local Vectorized'
# iterative_df['Approach'] = 'Local Iterative'
#
# # Concatenate the three dataframes
# concat_df = pd.concat([global_df, local_df, iterative_df], axis=0)
#
# # Reshape the dataframe for box plot
# melted_df = pd.melt(concat_df, id_vars=['Approach', 'Function Number'], value_vars=['Mean Error', 'Standard Deviation', 'Mean Time'], var_name='Metric', value_name='Value')
#
# # Generate box plots
# plt.figure(figsize=(12, 6))
# plt.title('Comparison of Approaches for Each Function Result')
# plt.xticks(rotation=45)
#
# # Create box plot
# boxplot = sns.boxplot(data=melted_df, x='Function Number', y='Value', hue='Approach')
#
# # Add labels and legend
# boxplot.set_xlabel('Function Number')
# boxplot.set_ylabel('Value')
# boxplot.legend(title='Approach')
#
# # Show the plot
# plt.show()

#
# # SCATTER PLOT
# # Add an "Approach" column to each DataFrame
# global_df['Approach'] = 'Global Vectorized'
# local_df['Approach'] = 'Local Vectorized'
# iterative_df['Approach'] = 'Local Iterative'
#
# # Concatenate the three dataframes
# concat_df = pd.concat([global_df, local_df, iterative_df], axis=0)
#
# # Create the scatter plot
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=concat_df, x='Mean Time', y='Mean Error', hue='Approach', palette='Set1')
#
# # Set plot labels and title
# plt.xlabel('Mean Time')
# plt.ylabel('Mean Error')
# plt.title('Comparison of Mean Error vs. Mean Time across Approaches')
#
# # Show the plot
# plt.show()