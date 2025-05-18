## ✨ Airline Passenger Satisfaction Predictor - Ai Project✨

An end-to-end Ai project that predicts airline passenger satisfaction using real-world travel survey data.It combines data preprocessing, machine learning, visualization, and a polished GUI using Tkinter.

---

### 🚀 Project Overview

This project aims to classify airline passengers as **Satisfied** or **Neutral/Dissatisfied** based on features related to their flight experience.It includes:

- 🔎 **Exploratory Data Analysis (EDA)** with Seaborn & Matplotlib

- 🧼 **Data cleaning** (handling missing values, encoding, outlier removal)

- 🔁 **Feature scaling** using StandardScaler

- 🧠 **Model training** using:

  - XGBoost (final model)
  - Logistic Regression
  - Random Forest
  - SVM
  - KNN
  - Decision Tree
  - Gradient Boosting
  - Neural Network (TensorFlow)

- 📈 **Evaluation** using Accuracy, Precision, Recall, F1 Score, ROC-AUC

- 🎯 Selection of best-performing model (XGBoost)

- 🖼️ **Heatmaps** and performance visualizations for comparison

---

### 🧩 Technologies Used

- Python (pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, tensorflow)
- Tkinter (for GUI)
- Jupyter Notebook (for analysis and development)

---

### 📦 How to Run

1. Clone the repo
    ```bash
   git clone https://github.com/Yomna-Mahsoob/Airline_Passanger_satisfication
   ```

2. Run the Jupyter notebook to train and save the model:
   ```bash
   python airline_model.py
   ```
3. Launch the GUI:
   ```bash
   python GUI.py
   ```

---

### 📁 Files Structure

```
📂 project-folder
├── airline_model.py       # Full analysis, model training, evaluation
├── GUI.py          # GUI app for predictions
├── xgb_model.pkl             # Saved XGBoost model
├── scaler.pkl                # Saved StandardScaler
```

---

### 💡 Inspiration

This project is an example of turning a machine learning model into a user-friendly product.It's perfect for those learning how to bridge the gap between **data science** and **real-world usability**.

---
### 🎀 The GUI

The project features a custom-designed Tkinter GUI with:

- Pink-themed visual design 🎨
- Airplane icon integration ✈️
- Friendly input fields for passenger details
- One-click prediction using the trained XGBoost model
- Helpful pop-up messages for input errors and results

---
### 🖼️ GUI Preview

Here’s a quick preview of the interface:

![GUI Preview](https://github.com/Yomna-Mahsoob/Airline_Passanger_satisfication/blob/main/GUI.jpeg)

---
### 🩷 Made BY:
- Thoraya Khaled
- Yomna EL-Kobesy
- Roquia Muhammad
- Roquia Rady
- Shahd Mazen
- Shrouk Bekheet
- Zainab Gamal
