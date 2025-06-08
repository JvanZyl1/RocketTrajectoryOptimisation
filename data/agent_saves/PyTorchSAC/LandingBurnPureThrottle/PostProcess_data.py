import os
import csv
import math
from collections import defaultdict

# Base directory for all PyTorchSAC runs
base_dir = "data/agent_saves/PyTorchSAC"

# Files to add to .gitignore
gitignore_entries = set()

# Get all subdirectories (different environments)
env_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
print(f"Found {len(env_dirs)} environment directories")

total_processed = 0

# Process each environment directory
for env_dir in env_dirs:
    env_path = os.path.join(base_dir, env_dir)
    print(f"\nProcessing environment: {env_dir}")
    
    # Get all run directories within this environment
    run_dirs = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
    print(f"Found {len(run_dirs)} run directories")
    
    # Process each run directory
    for run_dir in run_dirs:
        run_path = os.path.join(env_path, run_dir)
        print(f"  Processing run: {run_dir}")
        
        # Find all CSV files for this run
        csv_files = []
        
        # Look for learning_stats folder
        learning_stats_path = os.path.join(run_path, "learning_stats")
        if os.path.exists(learning_stats_path):
            for file in os.listdir(learning_stats_path):
                if file.endswith("sac_learning_stats.csv"):
                    csv_files.append(os.path.join(learning_stats_path, file))
        
        # Look for agent_saves folder
        agent_saves_path = os.path.join(run_path, "agent_saves")
        if os.path.exists(agent_saves_path):
            for file in os.listdir(agent_saves_path):
                if file.endswith("sac_pytorch_learning_stats.csv"):
                    csv_files.append(os.path.join(agent_saves_path, file))
            
        if not csv_files:
            print(f"    No CSV files found in {run_dir}")
            continue
            
        # Process each CSV file
        for csv_file in csv_files:
            try:
                # Add the original file to gitignore entries
                rel_path = os.path.relpath(csv_file)
                gitignore_entries.add(rel_path)
                
                # Read the CSV file
                with open(csv_file, 'r') as file:
                    csv_reader = csv.reader(file)
                    headers = next(csv_reader)  # Get the header row
                    data = list(csv_reader)  # Read all data rows
                
                if not data:
                    print(f"    Empty file: {csv_file}")
                    continue
                
                # Group by every 10 steps and calculate the mean
                groups = defaultdict(list)
                
                # Check if we have numeric data in the rows
                try:
                    # Try to convert first column to numbers
                    steps = [float(row[0]) for row in data]
                    
                    # Group by step/10 (rounded down)
                    for i, row in enumerate(data):
                        step = float(row[0])
                        group_key = math.floor(step / 10)
                        groups[group_key].append([
                            float(val) if val and (val.replace('-', '', 1).replace('.', '', 1).isdigit() or 
                                                  val.lower() in ('nan', 'inf', '-inf')) 
                            else 0 for val in row
                        ])
                
                except (ValueError, IndexError):
                    # If step column doesn't exist or isn't numeric, group by row index
                    for i, row in enumerate(data):
                        group_key = math.floor(i / 10)
                        groups[group_key].append([
                            float(val) if val and (val.replace('-', '', 1).replace('.', '', 1).isdigit() or 
                                                  val.lower() in ('nan', 'inf', '-inf')) 
                            else 0 for val in row
                        ])
                
                # Calculate the mean for each group
                reduced_data = []
                for group_key, rows in sorted(groups.items()):
                    if not rows:
                        continue
                        
                    # Calculate the average for each column
                    avg_row = []
                    for col in range(len(rows[0])):
                        col_sum = sum(row[col] for row in rows)
                        col_avg = col_sum / len(rows)
                        avg_row.append(col_avg)
                    
                    reduced_data.append(avg_row)
                
                # Create the new filename
                dir_path = os.path.dirname(csv_file)
                file_name = os.path.basename(csv_file)
                new_file_name = os.path.splitext(file_name)[0] + "_reduced.csv"
                new_file_path = os.path.join(dir_path, new_file_name)
                
                # Save the reduced data
                with open(new_file_path, 'w', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow(headers)  # Write the header row
                    csv_writer.writerows(reduced_data)  # Write the data rows
                
                # Check if the reduced file is larger than 80MB
                reduced_file_size_mb = os.path.getsize(new_file_path) / (1024 * 1024)
                if reduced_file_size_mb > 80:
                    # Add the reduced file to gitignore if it's larger than 80MB
                    rel_reduced_path = os.path.relpath(new_file_path)
                    gitignore_entries.add(rel_reduced_path)
                    print(f"    Added to .gitignore (>80MB): {os.path.basename(new_file_path)}")
                
                print(f"    Processed: {os.path.basename(csv_file)} → {new_file_name}")
                total_processed += 1
                
            except Exception as e:
                print(f"    Error processing {csv_file}: {str(e)}")

# Update .gitignore file
gitignore_path = '.gitignore'
existing_entries = set()

# Read existing .gitignore entries
if os.path.exists(gitignore_path):
    with open(gitignore_path, 'r') as file:
        existing_entries = set(line.strip() for line in file if line.strip() and not line.strip().startswith('#'))

# Add new entries
with open(gitignore_path, 'a') as file:
    file.write("\n# Automatically added by PostProcess_data.py\n")
    new_entries = sorted(gitignore_entries - existing_entries)
    for entry in new_entries:
        file.write(f"{entry}\n")
    print(f"\nAdded {len(new_entries)} new entries to .gitignore")

print(f"\nProcessing complete! Processed {total_processed} files.")
