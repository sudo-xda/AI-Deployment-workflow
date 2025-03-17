from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


model_filename = 'patia_house_modelv2.sav'
with open(model_filename, 'rb') as model_file:
    loaded_model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        year = int(request.form['year'])

        
        prediction = loaded_model.predict(np.array([[year]]))

        return render_template('res.html', prediction=prediction[0], year=year)

    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)
