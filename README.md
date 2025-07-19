# stock-prophet
Stock Price Movement Classifier using Technical Indicators &amp; Machine Learning

📈 Stock Price Movement Prediction Using Machine Learning

By Parin Thaokar

🔍 Overview
This project focuses on predicting whether a stock’s price will increase in the next 5 trading days using historical stock data and technical indicators. Leveraging a Random Forest Classifier, I built a binary classification model enhanced by hyperparameter tuning and supported with data visualizations.
The goal was to explore the intersection of finance and machine learning by analyzing real-world stock data and making short-term price direction predictions.
🧠 Problem Statement
The stock market is dynamic and influenced by countless factors. However, certain technical indicators provide insights into momentum, volatility, and price trends. This project explores whether machine learning can use these indicators to classify short-term future price movement as either up or down.
📁 Data Source
Source: Yahoo Finance (via finance library)
Stock Input: User-defined (e.g., AAPL, TSLA, MSFT)
Time Period: January 2020 – December 2024
Target Variable: Whether the stock’s price increases after 5 days
🧰 Tools & Technologies
Languages: Python
Libraries: pandas, finance, matplotlib, sklearn, GridSearchCV
Model: Random Forest Classifier
IDE: VS Code
📊 Technical Indicators Used
Indicator 					      Description
Daily Return
% change in price per day
MA5 / MA20/ MA50
Moving Averages over 5,20, and 50 days
Momentum(20)
Price Momentum over a 20-day lookback
Volatility
Rolling Standard Deviation of daily returns (20-day Window)
RSI
Relative Strength Index to gauge overbought/oversold conditions
Volume MA5/ MA20
Rolling average of volume over 5 and 20 days


🧪 Model Development
Data Downloading using yfinance
Feature Engineering with technical indicators
Data Cleaning (drop NAs after rolling calculations)
Train/Test Split using train_test_split()
Model Training with RandomForestClassifier
Hyperparameter Tuning using GridSearchCV
Prediction & Evaluation on test set
⚙️ Hyperparameter Tuning
I used a grid search to optimize:
Number of trees (n_estimators)
Max tree depth (max_depth)
Minimum samples per leaf and split
This helped boost the model’s accuracy while avoiding overfitting.
📈 Visualizations
Visuals were created with matplotlib for better interpretability:
Stock Price with MA5 / MA20 / MA50
RSI with Overbought (70) / Oversold (30) Zones
Volume Over Time
Prediction vs Actual Labels (Sample of 50)
🎯 Results
Accuracy (best model): ~80%
Best Parameters:{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
Metrics:
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)

🚀 How to Run the Project
Clone the repo or copy the code into a Jupyter Notebook or Python file
Install required libraries:
pip install pandas yfinance matplotlib scikit-learn
Run the script
Enter any stock ticker when prompted (e.g., AAPL, TSLA)
View predictions, accuracy, and graphs
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

	
