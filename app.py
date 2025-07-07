import streamlit as st
import pandas as pd
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Load the trained model
model = joblib.load("models/rf_model.pkl")
expected_features = model.feature_names_in_.tolist()

st.set_page_config(page_title="Tool Wear Prediction", layout="wide")
st.title("🔧 Tool Wear Prediction for CNC Machining")

# Sidebar for inputs
st.sidebar.header("📌 Enter CNC Parameters")
wear_threshold = st.sidebar.slider("🔧 Tool Wear Threshold (µm)", 0, 80, 60)

def get_user_input():
    cutting_speed = st.sidebar.number_input("Cutting Speed (m/min)", min_value=0.0, max_value=10000.0, value=120.0)
    feed_rate = st.sidebar.number_input("Feed Rate (mm/rev)", min_value=0.0, max_value=1.0, value=0.3)
    depth_of_cut = st.sidebar.number_input("Depth of Cut (mm)", min_value=0.0, max_value=5.0, value=1.5)
    spindle_speed = st.sidebar.number_input("Spindle Speed (RPM)", min_value=0, max_value=10000, value=1500)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0, max_value=300, value=80)
    tool_time = st.sidebar.number_input("Tool Usage Time (min)", min_value=0, max_value=1000, value=30)
    vibration = st.sidebar.number_input("Vibration (mm/s²)", min_value=0.0, max_value=1.0, value=0.02)
    material = st.sidebar.selectbox("Material Type", ["Aluminum", "Titanium", "Steel"])
    material_encoded = {"Aluminum": 0, "Titanium": 1, "Steel": 2}[material]

    input_data = {
        "Cutting Speed (m/min)": cutting_speed,
        "Feed Rate (mm/rev)": feed_rate,
        "Depth of Cut (mm)": depth_of_cut,
        "Spindle Speed (RPM)": spindle_speed,
        "Temperature (°C)": temperature,
        "Tool Usage Time (min)": tool_time,
        "Vibration (mm/s²)": vibration,
        "Material Type": material_encoded
    }

    df = pd.DataFrame([input_data], columns=expected_features)
    return df


input_df = get_user_input()
st.subheader("📋 Input Parameters")
st.dataframe(input_df)

# Prediction block
if st.button("📈 Predict Tool Wear"):
    try:
        prediction = model.predict(input_df)[0]
        st.metric(label="🔧 Predicted Tool Wear", value=f"{prediction:.2f} µm")

        if prediction >= wear_threshold:
            st.error("🔴 Tool wear limit exceeded! Replace immediately.")
        else:
            st.success("✅ Tool is in usable condition.")

        # Save prediction history
        input_df["Predicted Tool Wear (µm)"] = prediction
        input_df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not os.path.exists("results.csv"):
            input_df.to_csv("results.csv", index=False)
        else:
            input_df.to_csv("results.csv", mode="a", header=False, index=False)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# Feature Importance
st.subheader("🔍 Feature Importance")
try:
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = importances.argsort()
    ax.barh([expected_features[i] for i in sorted_idx], importances[sorted_idx], color='skyblue')
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not plot feature importance: {e}")

# Safe loader for results.csv
def load_results():
    if not os.path.exists("results.csv") or os.path.getsize("results.csv") == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv("results.csv")
    except Exception as e:
        st.warning(f"⚠️ Corrupted results file: {e}. Resetting...")
        os.remove("results.csv")
        return pd.DataFrame()

# Show data insights
if st.checkbox("📊 Show Data Insights"):
    df = load_results()
    if not df.empty:
        st.markdown("### 🧾 Tool Wear Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Predicted Tool Wear (µm)"], bins=30, kde=True, ax=ax1, color='purple')
        ax1.set_xlabel("Tool Wear (µm)")
        ax1.set_title("Distribution of Predicted Tool Wear")
        st.pyplot(fig1)

        st.markdown("### 🔁 Correlation Matrix")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        try:
            corr = df.drop(columns=["Predicted Tool Wear (µm)", "Timestamp"]).corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)
        except:
            st.warning("Correlation matrix could not be calculated due to missing values.")
    else:
        st.info("No prediction data to show insights.")

# Show recent predictions
with st.expander("📁 Recent Predictions"):
    df = load_results()
    if not df.empty:
        st.dataframe(df.sort_values("Timestamp", ascending=False).head(10))
        st.download_button("⬇️ Download All Results", df.to_csv(index=False), file_name="tool_wear_results.csv")
    else:
        st.info("No saved results found.")
