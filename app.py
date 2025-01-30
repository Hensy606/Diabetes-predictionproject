from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetes_pedigree']),
            float(request.form['age'])
        ]
        prediction = model.predict([data])[0]
        result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
        return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
