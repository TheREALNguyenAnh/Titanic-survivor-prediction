import csv

# Load CSV data with default values for missing fields
def load_and_calculate_coefficients(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    # Skip the header
    header = data[0]
    data = data[1:]
    
    # Initialize coefficient list and count
    coefficients = [0] * 6  # We have 6 relevant features: Pclass, Sex, Age, SibSp, Parch, Fare
    count = 0
    
    # Default values
    default_age = 30
    default_sibsp = 0
    default_parch = 0
    default_fare = 0
    for row in data:
        survived = int(row[1])
        if survived == 1:
            survived = float(1)
        else:
            survived = float(0)

        try:
            # Extract features with defaults
            pclass = float(row[2])
            sex = 1 if row[4] == 'male' else 0
            
            # Use default values for missing Age, SibSp, Parch, and Fare
            age = float(row[5]) if row[5] != '' else default_age
            sibsp = float(row[6]) if row[6] != '' else default_sibsp
            parch = float(row[7]) if row[7] != '' else default_parch
            fare = float(row[9]) if row[9] != '' else default_fare
            
            # Calculate coefficients as Survived / feature
            row_coeffs = [
                survived / pclass,
                survived / sex if sex != 0 else survived,  # Avoid dividing by zero
                survived / age,
                survived / sibsp if sibsp != 0 else survived,
                survived / parch if parch != 0 else survived,
                survived / fare if fare != 0 else survived
            ]
            
            # Sum coefficients for averaging
            coefficients = [coefficients[i] + row_coeffs[i] for i in range(len(coefficients))]
            count += 1
        except ValueError:
            # Ignore rows with missing or invalid data
            continue
    
    # Average the coefficients
    coefficients = [coeff / count for coeff in coefficients]
    
    # Save the coefficients to a CSV file
    with open(output_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
        writer.writerow(coefficients)
    
    print(f"Coefficients saved to {output_filename}")

# Run to calculate and save coefficients
load_and_calculate_coefficients('train.csv', 'coefficients.csv')
