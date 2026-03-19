# Define the number of reps and deer per rep
num_reps = 4
deer_per_rep = 9

# Generate the codes
codes = []
for rep in range(1, num_reps + 1):
    for deer in range(1, deer_per_rep + 1):
        code = f'R{rep}_D{deer}'
        codes.append(code)

# Print the codes
for code in codes:
    print(code)

import csv

# Write the codes to a CSV file
with open('Deer_codes.csv', 'w', newline='') as csvfile:
    
    writer = csv.writer(csvfile)
    for code in codes:
        writer.writerow([code])
