from flask import Flask, render_template, request, flash, redirect, url_for, session
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open('crop_recommendation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Crop image mapping
crop_images = {
    "Rice": "rice.jpeg",
    "Maize": "maize.jpeg",
    "Jute": "jute.jpeg",
    "Cotton": "cotton.jpeg",
    "Coconut": "coconut.jpeg",
    "Papaya": "papaya.jpeg",
    "Orange": "orange.jpeg",
    "Apple": "apple.jpeg",
    "Muskmelon": "muskmelon.jpeg",
    "Watermelon": "watermelon.jpeg",
    "Grapes": "grapes.jpg",
    "Mango": "mango.jpg",
    "Banana": "banana.jpeg",
    "Pomegranate": "pomegranate.jpeg",
    "Lentil": "lentil.jpeg",
    "Blackgram": "blackgram.jpeg",
    "Mungbean": "mungbean.jpeg",
    "Mothbeans": "mothbeans.jpeg",
    "Pigeonpeas": "pigeonpeas.jpeg",
    "Kidneybeans": "kidneybeans.jpeg",
    "Chickpea": "chickpea.jpeg",
    "Coffee": "coffee.jpeg"
}

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'  # Set a secret key for flash messages

# Define reasonable upper limits for input values
MAX_N = 100  # Maximum nitrogen content
MAX_P = 100  # Maximum phosphorus content
MAX_K = 100  # Maximum potassium content
MAX_TEMP = 50  # Maximum temperature in Celsius
MAX_HUMIDITY = 100  # Maximum relative humidity in %
MAX_PH = 14  # Maximum pH value
MAX_RAINFALL = 500  # Maximum rainfall in mm

@app.route('/')
def index():
    # Retrieve result and crop image from session, if available
    result = session.pop('result', None)
    crop_image = session.pop('crop_image', None)
    return render_template("index.html", result=result, crop_image=crop_image)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get input data from the form
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Validate input data
        if (N <= 0 or P <= 0 or K <= 0 or 
            temperature <= 0 or humidity <= 0 or 
            ph <= 0 or rainfall <= 0):
            flash("Input values must be greater than zero.", "error")
            return redirect(url_for('index'))
        
        if (N > MAX_N or P > MAX_P or K > MAX_K or
            temperature > MAX_TEMP or humidity > MAX_HUMIDITY or
            ph > MAX_PH or rainfall > MAX_RAINFALL):
            flash("Input values must be within reasonable limits.", "error")
            return redirect(url_for('index'))

        # Prepare input for the model
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get the crop image based on prediction
        crop_image = crop_images.get(prediction.title(), "default.jpeg")

        # Store result and image in session
        session['result'] = f"The recommended crop is: {prediction}"
        session['crop_image'] = crop_image

        # Redirect to main page to display result
        return redirect(url_for('index'))

    except ValueError:
        flash("Please enter valid numeric values.", "error")
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
