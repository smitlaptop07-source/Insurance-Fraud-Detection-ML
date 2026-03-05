from flask import Flask, request, render_template
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

model  = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            float(request.form['policy_annual_premium']),
            float(request.form['umbrella_limit']),
            float(request.form['capital_gains']),
            float(request.form['capital_loss']),
            float(request.form['total_claim_amount']),
            float(request.form['bodily_injuries']),
            float(request.form['witnesses']),
            float(request.form['incident_hour_of_the_day']),
            float(request.form['number_of_vehicles_involved'])
        ]
        features_array  = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction      = model.predict(features_scaled)

        if prediction[0] == 1:
            result = "FRAUD DETECTED - This claim is likely fraudulent!"
            color  = "red"
        else:
            result = "LEGITIMATE CLAIM - This claim appears genuine."
            color  = "green"
    except Exception as e:
        result = f"Error: {str(e)}"
        color  = "orange"

    return render_template('index.html', prediction_text=result, color=color)

if __name__ == '__main__':
    print("="*50)
    print("  Insurance Fraud Detection Web App")
    print("  Open: http://127.0.0.1:5000")
    print("="*50)
    app.run(debug=True)
