# House-Price-Predictor-India
"A Machine Learning-based House Price Predictor specifically designed for the Indian Real Estate market. Uses Linear Regression to estimate property values in Lakhs/Crores based on Area, BHK, and Age."
# 🏠 Smart India House Price Predictor

## 📌 Project Overview
This is a beginner-friendly Machine Learning project that solves a real-world problem: **Estimating property prices in India.** In a volatile market, buyers and sellers often struggle to find a fair price. This tool uses historical data patterns (Area, BHK, and Property Age) to provide an instant, data-driven valuation in Indian currency (Lakhs & Crores).

## 🚀 Key Features
- **India-Centric:** Handles prices in Lakhs and Crores.
- **Multi-Factor Analysis:** Considers Square Footage, BHK, and Building Age.
- **Interactive:** Allows users to input their own data for instant predictions.
- **Accurate Evaluation:** Uses Mean Absolute Error (MAE) to monitor prediction quality.

## 🛠️ Technology Stack
- **Language:** Python 3.x
- **Libraries:** - `Pandas` - For data manipulation.
  - `NumPy` - For numerical operations.
  - `Scikit-Learn` - For the Linear Regression model.
  - `Matplotlib` - For data visualization.

## 📊 How It Works
1. **Data Creation:** Generates a synthetic dataset of 300+ Indian properties.
2. **Preprocessing:** Cleans and prepares the data for the model.
3. **Training:** The model learns the relationship: 
   *Price increases with Area & BHK, but decreases with Age.*
4. **Testing:** Evaluates accuracy on unseen data (20% test split).

## 🏃 How to Run
1. Clone this repository or download the `.py` file.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
