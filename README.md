# 🏘️ HDB Resale Price Predictor – MLDP Final Project

![Python](https://img.shields.io/badge/Built%20With-Python%20%7C%20CatBoost%20%7C%20Streamlit-blue)
![Deployment](https://img.shields.io/badge/Deployed-On%20Streamlit-green)
This project was developed as part of the **Machine Learning for Developers Project (MLDP)** module. It focuses on building and deploying a machine learning model to predict **Singapore HDB resale prices** using publicly available housing data. The best model, after comparative evaluation and tuning, was deployed using **Streamlit** to provide a simple web interface for price predictions.

---

## 📊 Objective

To build a robust and accurate machine learning model capable of predicting resale prices for HDB flats based on flat type, location, floor area, proximity to MRT/CBD, and other housing features.

---

## 🧰 Tools & Libraries Used

- **Python**, **Pandas**, **NumPy**
- **Scikit-learn**
- **CatBoost**, **XGBoost**, **Random Forest**
- **Matplotlib**, **Seaborn**
- **Streamlit** (for web deployment)
- **Joblib** (for model serialisation)

---

## 📈 Workflow Overview

### 1. 🧹 Data Preprocessing
- Cleaned dataset (`hdb-resale-price.csv`)
- Dropped irrelevant columns (e.g., block, postal code, latitude/longitude)
- Converted dates and extracted temporal features
- Outlier filtering using 95th percentile
- One-hot encoding for categorical variables
- Feature scaling with `StandardScaler`

### 2. 🏗️ Feature Engineering
- Property age: `transaction_year - lease_commence_year`
- Binary flag: `large_floor_area` (over 100 sqm)
- Ratio features: MRT to CBD distance ratio

### 3. 🤖 Models Evaluated
| Model              | Tuned | Notes |
|-------------------|-------|-------|
| Random Forest      | ✅    | Baseline performance |
| Decision Tree      | ✅    | Weakest performer |
| Gradient Boosting  | ✅    | Good generalization |
| **CatBoost**       | ✅✅   | ⭐ Best performer |
| XGBoost            | ✅    | Also strong |

---

## 🏆 Best Model: CatBoost Regressor

After training and tuning multiple models, **CatBoost** emerged as the best performer with:

- **MAE**: ~28,000 SGD  
- **RMSE**: ~41,000 SGD  
- **R² Score**: ~0.91  
- **Training Time**: Fast and robust  
- **Hyperparameters Tuned**: iterations, depth, learning_rate

The final model was saved using `joblib` and deployed through a Streamlit web application.

---

## 🚀 Streamlit Web Application

A user-friendly interface was built using Streamlit, allowing users to:

- Input flat attributes (e.g., town, flat type, floor area)
- Instantly get a predicted resale price
- Run on local or public cloud (e.g., Streamlit Cloud)

> _Note: App is powered by the serialized `house_price_prediction_catboost.pkl` model._

---

## 🧪 Evaluation Techniques

- 5-fold Cross-Validation
- Learning Curves
- MAE, MSE, RMSE, R² Metrics
- Actual vs Predicted Plot
- Feature Scaling and Residual Distribution

---

## 🧠 How to Run This Project

### 🔧 Install dependencies

```bash
pip install -r requirements.txt
```

### 🧪 To run the notebook

```bash
jupyter notebook MLDP_Project.ipynb
```

### 🌐 To run the Streamlit app

```bash
streamlit run streamlit_app.py
```

---

## 📁 File Structure

```
.
├── MLDP_Project.ipynb                # Full modeling workflow
├── house_price_prediction_catboost.pkl  # Serialized best model
├── streamlit_app.py                  # Streamlit app frontend
├── hdb-resale-price.csv              # Source dataset
├── requirements.txt                  # All necessary libraries
```

---

## 👤 Author

**Moanish Ashok Kumar**  
Applied AI Final Year Student  
🔗 [LinkedIn](https://www.linkedin.com/in/moanish-ashok-kumar-086978272/)

---

> 🏠 This project aims to combine real-world housing data, machine learning, and frontend simplicity to create a useful tool for both citizens and data practitioners in Singapore.
