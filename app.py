# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import numpy as np
import pickle

# Start Flask App
app = Flask(__name__)

# LOAD PRE-TRAINED MODEL BY THE MODEL IPYNB

try:
    with open('model.pkl', 'rb') as file:
        saved_data = pickle.load(file)
    
    # Extract the tools
    model = saved_data['model']
    scaler = saved_data['scaler']
    imputer = saved_data['imputer']
    print("[SUCCESS] Pre-trained model.pkl loaded successfully!")
except FileNotFoundError:
    print("[ERROR] model.pkl not found. Please run model.ipynb first.")
    exit()

# FLASK ROUTING 
@app.route('/', methods=['GET'])
def welcome():
    """Renders the initial welcome/landing page."""
    return render_template('welcome.html')

@app.route('/analysis', methods=['GET'])
def form():
    """Renders the main data entry form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, processes data, and returns the prediction."""
    if request.method == 'POST':
        # get the user's name to display on the results page
        user_name = request.form['user_name']

        # get all numerical inputs from the HTML form
        age = float(request.form['age'])
        experience_years = float(request.form['experience_years'])
        daily_work_hours = float(request.form['daily_work_hours'])
        sleep_hours = float(request.form['sleep_hours'])
        caffeine_intake = float(request.form['caffeine_intake'])
        bugs_per_day = float(request.form['bugs_per_day'])
        commits_per_day = float(request.form['commits_per_day'])
        meetings_per_day = float(request.form['meetings_per_day'])
        screen_time = float(request.form['screen_time'])
        exercise_hours = float(request.form['exercise_hours'])
        stress_level = float(request.form['stress_level'])

        # make input array in the exact order of the features list
        input_data = np.array([[
            age, experience_years, daily_work_hours, sleep_hours,
            caffeine_intake, bugs_per_day, commits_per_day,
            meetings_per_day, screen_time, exercise_hours, stress_level
        ]])

        # Apply the exact same preprocessing to the user's input
        input_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_imputed)

        # make Prediction
        prediction = model.predict(input_scaled)[0]

        # make the output message and styling based on the result
        if prediction == 1:
            status = "System Classification: Robot"
            message = "Warning: High Burnout! You are working like a robot and not getting enough rest."
            alert_class = "alert-secondary"
        else:
            status = "System Classification: Human"
            message = "Great news! You are living like a human, with a healthy balance of work and time to recharge."
            alert_class = "alert-success"

        # Pass the results and the updated top 5 feature inputs back to the frontend
        return render_template('result.html',
                               name=user_name,
                               status=status,
                               message=message,
                               alert_class=alert_class,
                               stress=stress_level,
                               work=daily_work_hours,
                               screen=screen_time,
                               bugs=bugs_per_day,
                               meetings=meetings_per_day)

# execute
if __name__ == '__main__':
    # Run the application on localhost, port 5000
    app.run(debug=True, port=5000)