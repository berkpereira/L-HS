"""
This script's purpose is simply to find all the CUTEst problems remotely
applicable to our purposes.
"""
import io
from collections import Counter
import sys
import numpy as np
import pycutest
import os
import json

os.system('clear')

def save_matching_problems_to_file(matching_problems, filename):
    # Check if the file exists
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            existing_problems = json.load(file)
    else:
        existing_problems = []

    # Combine the existing problems with the new ones, avoiding duplicates
    combined_problems = list(set(existing_problems + matching_problems))

    with open(filename, 'w') as file:
        json.dump(combined_problems, file)

# Function to read the list from a file
def load_matching_problems_from_file(filename):
    with open(filename, 'r') as file:
        matching_problems = json.load(file)
    return matching_problems

def check_problems_within_range(probs, lower_bound=50, upper_bound=100):
    matching_problems = []

    for p_name in probs:
        print(f'CURRENT list: {matching_problems}')
        # Import the problem
        print(f'TRYING to look at {p_name}')
        try:
            p = pycutest.import_problem(p_name)
        except:
            print(f'SKIPPED {p_name}')
            print()
            continue
        
        # Capture the printed output of available parameters
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            pycutest.print_available_sif_params(p_name)
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()
        lines = output.split('\n')
        
        # Check for adjustable 'N' parameters within the desired range
        found_adjustable = False
        for line in lines:
            if 'N =' in line:
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        n_value = int(part)
                        if lower_bound <= n_value <= upper_bound:
                            matching_problems.append(f"{p_name}_n{n_value}")
                            found_adjustable = True
        
        # If no adjustable 'N' parameter found, check the dimension
        if not found_adjustable:
            if lower_bound <= p.n <= upper_bound:
                matching_problems.append(f"{p_name}_n{p.n}")

    return matching_problems

################################################################################

# probs = pycutest.find_problems(constraints='unconstrained',
#                                regular=True,
#                                degree=[2, 1000])

# for some reason the loop always gets stuck on this one
# probs.remove('BA-L52LS')

FILENAME = 'problems/matching_problems.json'

# LOWER_DIM = 1
# UPPER_DIM = 500

# matching_problems = check_problems_within_range(probs=probs,lower_bound=LOWER_DIM,
#                                                 upper_bound=UPPER_DIM)

# Save (append) the list to a file
# save_matching_problems_to_file(matching_problems, FILENAME)

# # Read the list back from the file
# loaded_problems = load_matching_problems_from_file(FILENAME)
# print("Loaded problems:", loaded_problems)
# # Strip the suffixes from the problem names
# base_names = [name.split('_')[0] for name in loaded_problems]

# # Count the occurrences of each problem name
# name_counts = Counter(base_names)

# # Sort the problem names by their frequency in descending order
# sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)

# # Print the sorted problem names
# for name, count in sorted_names:
#     print(f"{name}: {count}")