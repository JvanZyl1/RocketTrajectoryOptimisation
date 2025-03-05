from src.envs.env_setup.main_sizing import size_rocket
import csv
size_rocket()

# Read the csv data/sizing_results.csv
# Has format: Key, Unit, Value
# Save the values in a dictionary
sizing_results = {}
with open('data/sizing_results.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sizing_results[row[0]] = row[2]

# Print the dictionary
for key, value in sizing_results.items():
    print(f'{key}: {value}')