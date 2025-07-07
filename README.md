# Tool-Wear-Prediction-for-CNC-Machining
A machine learning-based web app that predicts tool wear from CNC parameters and alerts whether the tool is still usable, helping optimize performance and reduce downtime.


# 🔧 Tool Wear Prediction for CNC Machining

This project uses a **machine learning model** to predict tool wear in CNC machining processes based on various input parameters such as cutting speed, feed rate, spindle speed, temperature, and material type.

The project is built using **Python**, **scikit-learn**, and **Streamlit** and includes:
- A trained Random Forest Regression model.
- A web interface for real-time predictions.
- Visualization of feature importance and prediction insights.
- Auto-saving of prediction history.

---

## 📂 Project Structure

```
tool_wear_app.py       # Main application file (training + Streamlit app)
models/
└── rf_model.pkl       # Trained machine learning model
results.csv            # Prediction history (auto-generated)
```

---

## 🚀 How to Run the App

1. **Clone the repository** (or download the file):
   ```bash
   git clone https://github.com/your-username/tool-wear-prediction.git
   cd tool-wear-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run tool_wear_app.py
   ```

---

## 🧪 Features

- 📈 Predict CNC tool wear instantly.
- 📊 Visualize feature importance and data insights.
- 🗃️ Save and export prediction history.
- 🎛️ Input values via number fields and dropdowns.

---

## 🧠 Model Details

- Model: **Random Forest Regressor**
- Training Data: **Synthetic CNC parameter dataset (5,000 rows)**
- Target Variable: **Tool Wear (in micrometers)**

---

## 🖥️ Input Parameters

- Cutting Speed (m/min)
- Feed Rate (mm/rev)
- Depth of Cut (mm)
- Spindle Speed (RPM)
- Temperature (°C)
- Tool Usage Time (min)
- Vibration (mm/s²)
- Material Type: Aluminum, Titanium, Steel

---

## 📌 Output

- 🔧 Predicted Tool Wear (µm)
- ✅ Tool Usable Condition / 🔴 Replacement Required

---

## 📎 Requirements

- Python 3.8+
- Libraries: `streamlit`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `numpy`

You can install them using:

```bash
pip install streamlit pandas scikit-learn joblib matplotlib seaborn numpy
```

---


## 🙌 Author

**Pramukh Prajapati**  
Tool Wear Prediction App · 2025

---
