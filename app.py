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
            float(request.form['policy_state']),
            float(request.form['policy_csl']),
            float(request.form['policy_deductable']),
            float(request.form['policy_annual_premium']),
            float(request.form['umbrella_limit']),
            float(request.form['insured_sex']),
            float(request.form['insured_education_level']),
            float(request.form['insured_occupation']),
            float(request.form['insured_hobbies']),
            float(request.form['insured_relationship']),
            float(request.form['capital_gains']),
            float(request.form['capital_loss']),
            float(request.form['incident_type']),
            float(request.form['collision_type']),
            float(request.form['incident_severity']),
            float(request.form['authorities_contacted']),
            float(request.form['incident_state']),
            float(request.form['incident_city']),
            float(request.form['incident_hour_of_the_day']),
            float(request.form['number_of_vehicles_involved']),
            float(request.form['property_damage']),
            float(request.form['bodily_injuries']),
            float(request.form['witnesses']),
            float(request.form['police_report_available']),
            float(request.form['total_claim_amount']),
            float(request.form['auto_year']),
            float(request.form['age']),
            float(request.form['incident_date']),
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
