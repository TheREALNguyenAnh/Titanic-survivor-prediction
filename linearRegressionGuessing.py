import csv

# Predict using coefficients and ensure output is 0 or 1
def predict_survived(input_filename, coefficients_filename, output_filename):
    # Load coefficients from the file
    with open(coefficients_filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        coefficients = [float(c) for c in next(reader)]
    
    # Default values for missing fields
    default_age = 30  # Default age if missing
    default_sibsp = 0  # Default siblings/spouses if missing
    default_parch = 0  # Default parents/children if missing
    default_fare = 0  # Default fare if missing

    predictions = []
    
    with open(input_filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    # Skip header
    header = data[0]
    data = data[1:]
    
    for row in data:
        try:
            # Extract features with defaults
            pclass = float(row[1])
            sex = 1 if row[3] == 'male' else 0
            age = float(row[4]) if row[4] != '' else default_age
            sibsp = float(row[5]) if row[5] != '' else default_sibsp
            parch = float(row[6]) if row[6] != '' else default_parch
            fare = float(row[8]) if row[8] != '' else default_fare
            
            # Calculate predicted survived value using the coefficients
            survived_pred = (
                coefficients[0] * pclass +
                coefficients[1] * sex +
                coefficients[2] * age +
                coefficients[3] * sibsp +
                coefficients[4] * parch +
                coefficients[5] * fare
            )
            
            
            if survived_pred >= 0.5:
                survived_pred = 1
            else:
                survived_pred = 0
            # Save PassengerId and predicted Survived value
            passenger_id = row[0]
            predictions.append([passenger_id, survived_pred])
        
        except ValueError:
            # Skip rows with missing or invalid data
            continue
    
    # Save the predictions to a CSV file
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PassengerId', 'Survived'])
        writer.writerows(predictions)
    
    print(f"Predictions saved to {output_filename}")

# Run to make predictions
predict_survived('test.csv', 'coefficients.csv', 'predictions.csv')
