import csv
import numpy as np

# Function to load data and solve for coefficients using sets of 6 datapoints
def solve_coefficients(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    # Skip the header
    header = data[0]
    data = data[1:]
    
    # Initialize list for coefficients and count
    coefficients_list = []
    
    # Default values for missing data
    default_age = 30
    default_sibsp = 0
    default_parch = 0
    default_fare = 0

    # Process data in sets of 6
    for i in range(0, len(data) - 6, 6):
        try:
            A = []
            b = []
            
            for j in range(6):
                row = data[i + j]
                
                # Extract features with defaults
                pclass = float(row[2])
                sex = 1 if row[4] == 'male' else 0
                age = float(row[5]) if row[5] != '' else default_age
                sibsp = float(row[6]) if row[6] != '' else default_sibsp
                parch = float(row[7]) if row[7] != '' else default_parch
                fare = float(row[9]) if row[9] != '' else default_fare
                
                # Construct the equation: survived = pclass * c1 + sex * c2 + ... + fare * c6
                A.append([pclass, sex, age, sibsp, parch, fare])
                b.append(float(row[1]))  # Survived column as output
                
            # Solve the system of equations Ax = b for the coefficients
            A = np.array(A)
            b = np.array(b)
            coefficients = np.linalg.solve(A, b)
            
            coefficients_list.append(coefficients)
        
        except np.linalg.LinAlgError:
            # If the matrix is singular or not solvable, skip this set
            continue
        except ValueError:
            # Skip rows with invalid or missing data
            continue
    
    # Average the coefficients across all sets
    coefficients_avg = np.mean(coefficients_list, axis=0)
    
    # Save the average coefficients to a CSV file
    with open(output_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
        writer.writerow(coefficients_avg)
    
    print(f"Averaged coefficients saved to {output_filename}")

# Run to solve for and save averaged coefficients
solve_coefficients('train.csv', 'coefficients.csv')
