# 📈 stock-prophet

**Stock Price Movement Classifier using Technical Indicators & Machine Learning**

---

## 🔍 Overview

This project aims to predict whether a stock’s price will increase over the next 5 trading days, utilizing historical stock data and technical indicators. Leveraging a Random Forest Classifier, I built a binary classification model that was enhanced through hyperparameter tuning and supported by data visualizations.

The goal was to explore the intersection of finance and machine learning by analyzing real-world stock data and making short-term price direction predictions.

---

## 🧠 Problem Statement

The stock market is dynamic and influenced by countless factors. However, certain technical indicators provide insights into momentum, volatility, and price trends. This project explores whether machine learning can use these indicators to classify short-term future price movement as either **up** or **down**.

---

## 📁 Data Source

- **Source:** Yahoo Finance (via `yfinance` library)  
- **Stock Input:** User-defined (e.g., AAPL, TSLA, MSFT)  
- **Time Period:** January 2020 – December 2024  
- **Target Variable:** Whether the stock’s price increases after 5 days

---

## 🧰 Tools & Technologies

- **Languages:** Python  
- **Libraries:** pandas, yfinance, matplotlib, scikit-learn, GridSearchCV  
- **Model:** Random Forest Classifier  
- **IDE:** VS Code  

---

## 📊 Technical Indicators Used

| Indicator         | Description                                  |
|-------------------|----------------------------------------------|
| Daily Return      | % change in price per day                    |
| MA5 / MA20 / MA50 | Moving Averages over 5, 20, and 50 days      |
| Momentum (20)     | Price Momentum over a 20-day lookback        |
| Volatility       | Rolling Standard Deviation of daily returns (20-day window) |
| RSI               | Relative Strength Index to gauge overbought/oversold conditions |
| Volume MA5 / MA20 | Rolling average of volume over 5 and 20 days |

---

## 🧪 Model Development

- Downloaded data using `yfinance`  
- Engineered features based on technical indicators  
- Cleaned data by dropping NaNs after rolling calculations  
- Train/Test split using `train_test_split()`  
- Trained `RandomForestClassifier`  
- Performed hyperparameter tuning with `GridSearchCV`  
- Predicted and evaluated model on test set  

---

## ⚙️ Hyperparameter Tuning

Grid search optimized these parameters:  

- Number of trees (`n_estimators`)  
- Max tree depth (`max_depth`)  
- Minimum samples per leaf and split (`min_samples_leaf`, `min_samples_split`)  

This helped boost model accuracy while reducing overfitting.

---

## 📈 Visualizations

- Stock Price with MA5 / MA20 / MA50  
- RSI with Overbought (70) / Oversold (30) zones  
- Volume over time  
- Prediction vs Actual labels (sample of 50)

---

## 🎯 Results

- **Accuracy (best model):** ~80%  
- **Best Parameters:**  
  ```python
  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}


## 🚀 How to Run the Project

1. Clone the repository or copy the code into a Jupyter Notebook or Python file.  
2. Install the required libraries:
   ```bash
   pip install pandas yfinance matplotlib scikit-learn
   
---


🌟 Future Improvements
Add a Streamlit web dashboard
Introduce sentiment analysis using news or Twitter
Experiment with XGBoost or deep learning (LSTM) models
Include a backtesting engine to simulate real strategies


👨‍💻 About Me
Parin Thaokar
Aspiring Data Scientist & Finance Enthusiast
📍 Data Science | Interested in AI + Finance
www.linkedin.com/in/parinthaokar  | https://github.com/parinthaokar | parinthaokar@gmail.com 

	
