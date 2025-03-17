from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


model_filename = 'patia_house_modelv2.sav'
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('indexv2.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = int(request.form['year'])
        
        #sqft = int(request.form['sqft'])
        sqft = request.form.get('sqft', '1') 
        sqft = int(sqft) if sqft.strip() else 1 

       
        predicted_price_per_sqft = loaded_model.predict([[year]])[0]
        total_price = predicted_price_per_sqft * sqft

        return render_template('resv2.html', prediction=total_price, year=year, sqft=sqft)

    except Exception as e:
        return f"Error: {e}"



if __name__ == '__main__':
    app.run(debug=True)
