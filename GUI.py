import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Features (adjusted to match your training data)
features = ['Age', 'Flight Distance', 'Inflight wifi service', 'Ease of Online booking',
            'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Checkin service', 'Inflight service',
            'Cleanliness']

# Initialize GUI
root = tk.Tk()
root.title("🎀 Airline Satisfaction Predictor 🎀")
root.configure(bg="#ffe6f0")

entries = {}

# Title label
title_label = tk.Label(root, text="✨ Airline Satisfaction Prediction ✨",
                       font=("Helvetica", 16, "bold"), fg="#d63384", bg="#ffe6f0")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Input fields
for i, feature in enumerate(features):
    label = tk.Label(root, text=feature, font=("Helvetica", 11), bg="#ffe6f0", anchor="w")
    label.grid(row=i+1, column=0, padx=10, pady=5, sticky='w')
    entry = tk.Entry(root, width=25, font=("Helvetica", 10))
    entry.grid(row=i+1, column=1, padx=10, pady=5)
    entries[feature] = entry

# Prediction function
def predict():
    input_data = []
    for feature in features:
        val = entries[feature].get()
        try:
            num = float(val)
            input_data.append(num)
        except ValueError:
            messagebox.showerror("Input Error", f"Please enter a valid numeric value for '{feature}'. You entered: '{val}'")
            return

    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)
    result = "💖 Satisfied 💖" if prediction[0] == 1 else "😐 Neutral or Dissatisfied"
    messagebox.showinfo("Prediction Result", f"Prediction: {result}")

# Predict button
predict_button = tk.Button(root, text="✨ Predict ✨", command=predict,
                           bg="#ff66b2", fg="white", font=("Helvetica", 12, "bold"))
predict_button.grid(row=len(features)+2, column=0, columnspan=2, pady=20)

# Start the GUI
root.mainloop()
