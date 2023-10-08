# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('C:/Users/Zaiba Farheen/PycharmProjects/pythonProject2/DataScienceEL/Datasets/iris.csv')

# Split the dataset into features and target
X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
mlp.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            # Get user input from the HTML form
            user_input = [float(request.form['sepal_length']),
                          float(request.form['sepal_width']),
                          float(request.form['petal_length']),
                          float(request.form['petal_width'])]

            # Make a prediction using the trained model
            result = mlp.predict([user_input])[0]
        except Exception as e:
            result = str(e)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
