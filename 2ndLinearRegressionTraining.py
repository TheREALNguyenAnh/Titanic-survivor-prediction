import csv
from sklearn.linear_model import LinearRegression

# Load data from CSV, handling missing values and categorical encoding
def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        data = []
        for row in reader:
            # Convert necessary fields and handle missing values
            pclass = float(row['Pclass'])
            sex = 1 if row['Sex'] == 'male' else 0
            age = float(row['Age']) if row['Age'] else 30  # Default age if missing
            sibsp = float(row['SibSp']) if row['SibSp'] else 0  # Default if missing
            parch = float(row['Parch']) if row['Parch'] else 0  # Default if missing
            fare = float(row['Fare']) if row['Fare'] else 0  # Default if missing
            survived = int(row['Survived'])
            data.append([pclass, sex, age, sibsp, parch, fare, survived])
        return data

# Load and split data
data = load_data('train.csv')
X = [row[:-1] for row in data]  # Features
y = [row[-1] for row in data]    # Target

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Save coefficients
coefficients = model.coef_
intercept = model.intercept_

# Save coefficients to a file
with open('coefficients_sklearn.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Intercept', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
    writer.writerow([intercept] + list(coefficients))

print(f"Coefficients saved to coefficients_sklearn.csv")

# Function to predict survival using the model and save predictions
def predict_survived(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        reader = csv.DictReader(f)
        predictions = []
        for row in reader:
            # Extract features and handle missing values
            pclass = float(row['Pclass'])
            sex = 1 if row['Sex'] == 'male' else 0
            age = float(row['Age']) if row['Age'] else 30
            sibsp = float(row['SibSp']) if row['SibSp'] else 0
            parch = float(row['Parch']) if row['Parch'] else 0
            fare = float(row['Fare']) if row['Fare'] else 0

            # Predict using the trained model
            survived_pred = model.predict([[pclass, sex, age, sibsp, parch, fare]])[0]
            predictions.append(1 if survived_pred >= 0.5 else 0)

    # Save predictions
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PassengerId', 'Survived'])
        with open(input_filename, 'r') as infile:
            reader = csv.DictReader(infile)
            for passenger_id, pred in zip(reader, predictions):
                writer.writerow([passenger_id['PassengerId'], pred])

    print(f"Predictions saved to {output_filename}")

# Run prediction function
predict_survived('test.csv', 'predictions_sklearn.csv')
