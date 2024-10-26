import csv
import numpy as np

# Function to compute least squares coefficients for the entire dataset
def least_squares_all_data(input_filename, output_filename):
    # Load data
    with open(input_filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    # Skip header
    header = data[0]
    data = data[1:]
    
    # Prepare matrices for Least Squares
    X = []
    y = []
    
    # Default values for missing data
    default_age = 30
    default_sibsp = 0
    default_parch = 0
    default_fare = 0

    # Extract features and target variable
    for row in data:
        try:
            # Extract features with defaults
            pclass = float(row[2])
            sex = 1 if row[4] == 'male' else 0
            age = float(row[5]) if row[5] != '' else default_age
            sibsp = float(row[6]) if row[6] != '' else default_sibsp
            parch = float(row[7]) if row[7] != '' else default_parch
            fare = float(row[9]) if row[9] != '' else default_fare

            # Append a row of features (adding 1 for the intercept term)
            X.append([1, pclass, sex, age, sibsp, parch, fare])

            # Append corresponding `Survived` value
            y.append(float(row[1]))
        
        except ValueError:
            # Skip rows with invalid data
            continue

    # Convert X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Apply the least squares formula: b = (X^T * X)^(-1) * X^T * y
    Xt = X.T  # Transpose of X
    XtX = np.dot(Xt, X)  # X^T * X
    XtX_inv = np.linalg.inv(XtX)  # (X^T * X)^(-1)
    Xt_y = np.dot(Xt, y)  # X^T * y
    coefficients = np.dot(XtX_inv, Xt_y)  # Final coefficients

    # Save coefficients to a CSV file
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Intercept', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
        writer.writerow(coefficients)
    
    print(f"Coefficients saved to {output_filename}")

# Run the least squares method and save coefficients
least_squares_all_data('train.csv', 'coefficients.csv')
