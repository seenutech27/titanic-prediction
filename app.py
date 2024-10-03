from flask import Flask, render_template, request
from model import TitanicModel
import pandas as pd

app = Flask(__name__)

# Initialize your Titanic model with both model and training data paths
titanic_model = TitanicModel('titanic_model.pkl', 'train.csv')  # Provide the path to your train.csv

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    data = {
        'Pclass': int(request.form['Pclass']),
        'Sex': request.form['Sex'],
        'Age': float(request.form['Age']),
        'SibSp': int(request.form['SibSp']),
        'Parch': int(request.form['Parch']),
        'Fare': float(request.form['Fare']),
        'Embarked': request.form['Embarked']
    }

    # Convert categorical variables to numerical
    data['Sex'] = 1 if data['Sex'] == 'male' else 0

    # Create a DataFrame for the model
    features = pd.DataFrame([data])

    # One-hot encoding for 'Embarked'
    embarked_dummies = pd.get_dummies(features['Embarked'], prefix='Embarked', drop_first=True)
    features = pd.concat([features.drop('Embarked', axis=1), embarked_dummies], axis=1)

    # Ensure the DataFrame columns match the model's training columns
    expected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    
    for col in expected_columns:
        if col not in features.columns:
            features[col] = 0  # Assign a default value (0) if the column is missing
    
    features = features[expected_columns]  # Reorder to match the expected input

    # Get prediction
    prediction = titanic_model.predict(features)

    # Convert prediction to a readable format (0 -> No, 1 -> Yes)
    result = 'Survived' if prediction[0] == 1 else 'Not Survived'

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
