# 🛠️ Mechanical Part Identifier

This is a Streamlit web app that classifies mechanical parts using a trained TensorFlow model.

## 🚀 Features

- Upload any image and get part prediction (Bolt, Bearing, Nut, Gear)
- Interactive confidence bar chart
- Sample images you can preview and download

## 📁 Folder Structure

- `model.savedmodel/`: Saved TensorFlow model
- `samples/`: Sample images for testing
- `app.py`: Streamlit app
- `requirements.txt`: Python dependencies

## ▶ Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
