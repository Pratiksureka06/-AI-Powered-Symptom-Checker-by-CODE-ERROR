from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
import random

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Set up OpenAI client
client = OpenAI(api_key=os.getenv('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'))

def generate_medical_data(max_retries=5, base_delay=1):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical expert. Generate a list of 30 common symptoms and 20 medical conditions."},
                    {"role": "user", "content": "Please provide a list of 30 common symptoms and 20 medical conditions. Format your response as a Python dictionary with keys 'symptoms' and 'conditions', each containing a list of strings."}
                ]
            )
            
            # Parse the response and extract the symptoms and conditions
            generated_data = eval(response.choices[0].message.content)
            return generated_data['symptoms'], generated_data['conditions']
        except Exception as e:
            if "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    delay = (base_delay * 2 ** attempt) + (random.randint(0, 1000) / 1000)
                    print(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Using fallback data.")
                    return generate_fallback_data()
            else:
                print(f"An error occurred: {e}")
                return generate_fallback_data()

def generate_fallback_data():
    symptoms = ["fever", "cough", "headache", "fatigue", "nausea"]
    conditions = ["Common Cold", "Flu", "Migraine", "Dehydration"]
    return symptoms, conditions

# Generate symptoms and conditions
symptoms, conditions = generate_medical_data()

# Create a simple dataset
num_samples = 5000
X = np.random.randint(2, size=(num_samples, len(symptoms)))
y = np.random.choice(conditions, size=num_samples)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Create a MultiLabelBinarizer for symptom encoding
mlb = MultiLabelBinarizer()
mlb.fit([symptoms])

# Define a mapping of conditions to suggested next steps
condition_steps = {
    "Common Cold": "Stay hydrated, rest, and consider over-the-counter cold medications.",
    "Flu": "Consult with a doctor, consider antiviral medications, and rest.",
    "Migraine": "Avoid triggers, rest in a dark room, and consider pain relief medications.",
    "Dehydration": "Increase fluid intake, and consult a doctor if severe.",
    # Add more conditions and their suggestions here
}

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms)

@app.route('/analyze', methods=['POST'])
def analyze():
    user_symptoms = request.json['symptoms']
    
    # Encode user symptoms
    X_user = mlb.transform([user_symptoms])
    
    # Predict probabilities for each condition
    probabilities = clf.predict_proba(X_user)[0]
    
    # Get top 3 conditions
    top_3_indices = probabilities.argsort()[-3:][::-1]
    results = [
        {"condition": conditions[i], "confidence": float(probabilities[i])}
        for i in top_3_indices
    ]
    
    # Prepare next steps based on conditions
    next_steps = [
        {"condition": results[i]['condition'], "steps": condition_steps.get(results[i]['condition'], "No suggestions available.")}
        for i in range(len(results))
    ]
    
    return jsonify({"conditions": results, "next_steps": next_steps})

if __name__ == '__main__':
    app.run(debug=True)
