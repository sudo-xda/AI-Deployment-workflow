import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd

with open('model_and_norm.pkl', 'rb') as file:
    loaded_model, Tr_Fn = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

selected_features = ['citric_acid', 'residual_sugar', 'pH', 'sulphates', 'alcohol']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        x_in = pd.DataFrame([{feature: float(request.form[feature]) for feature in selected_features}])
        
     
        norm = Tr_Fn.fit_transform(x_in)
        
     
        y_pred = loaded_model.predict(norm)[0]
        
        return render_template('res.html', prediction=y_pred)
    except Exception as e:
        return f"Error: {e}" 


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
       
        data = request.get_json()
        x_in = pd.DataFrame([{
            "citric acid": float(data['citric_acid']),
            "residual sugar": float(data['residual_sugar']),
            "pH": float(data['pH']),
            "sulphates": float(data['sulphates']),
            "alcohol": float(data['alcohol'])
        }])

      
        norm = Tr_Fn.fit_transform(x_in)
        x=norm.reshape(1,-1)

    
        y_pred = loaded_model.predict(x)

        print(f"Raw input: {data}")
        print(f"Transformed input: {x}")
        print(f"Prediction: {y_pred}")
        
        return jsonify({'prediction': float(y_pred)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5008, debug=True)
