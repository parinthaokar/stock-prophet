import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

ticker = input("Enter Stock Ticker Symbol (e.g. AAPL, MSFT, TSLA): ").upper()


df = yf.download(ticker,start = '2020-01-01', end = '2024-12-31', auto_adjust=False)

#Daily Return
df['Daily Return'] = df['Adj Close'].pct_change()


#Moving Average Over 5 and 50 days
df['MA5'] = df['Adj Close'].rolling(window=5).mean()
df['MA20'] = df['Adj Close'].rolling(window = 20).mean()
df['MA50'] = df['Adj Close'].rolling(window = 50).mean()

#Target
future_days = 5
df['Target'] = (df["Adj Close"].shift(-future_days)>df['Adj Close']).astype(int)

#Momentum
lookback_period = 20
df['Momentum'] = df['Adj Close'].pct_change(periods=lookback_period)

#Volitality
df['Volatility'] = df['Daily Return'].rolling(window=20).std()

# RSI (14-day)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI14'] = compute_rsi(df['Adj Close'], 14)

# Volume Moving Averages
df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

#Clean Data
df = df.dropna()

#Feature and Targets

x = df[['Daily Return', 'MA5', 'MA20', 'MA50', 'Momentum', 'Volatility', 'RSI14', 'Volume_MA5', 'Volume_MA20']]
y = df['Target']

X_train = x[df.index < '2023-12-31']
X_test = x[df.index >= '2023-12-31']

X_train, X_test, Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Model Tuning
param_grid = {'n_estimators' : [50,100,500], 'max_depth': [None, 5,10,20], 'min_samples_split': [2,5,10], 'min_samples_leaf': [1,2,4]}
model = RandomForestClassifier(random_state= 42,class_weight='balanced')
grid_search = GridSearchCV(estimator= model, param_grid=param_grid, cv = 5, scoring='accuracy', n_jobs=1, verbose = 2)

grid_search.fit(X_train,Y_train)

# Predict on test set
best_rf = grid_search.best_estimator_
predictions = best_rf.predict(X_test)


print(df[['Adj Close', 'Daily Return','MA5','MA20','MA50','Momentum','Volatility','Target']].tail(10))


# Evaluation
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy Score:", grid_search.best_score_)

print("Classification Report:\n", classification_report(Y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(Y_test, predictions))

#Moving Averages Graph
plt.figure(figsize=(14,6))
plt.plot(df['Adj Close'], label = 'Adjusted Close Price', color = 'black')
plt.plot(df['MA5'], label ="MA5", linestyle = 'dotted')
plt.plot(df['MA20'], label ="MA20", linestyle = 'dotted')
plt.plot(df['MA50'], label ="MA50", linestyle = 'dotted')
plt.title('Stock Prices with Moving Averages')
plt.title(ticker, fontsize = 10, y=1, loc='right')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.savefig("charts/MovingAverage")
plt.show()

#RSI Graph
plt.figure(figsize=(14,4))
plt.plot(df["RSI14"], label = 'RSI 14', color = 'purple')
plt.axhline(70, color = 'red', linestyle ='dotted', label = 'Overbought (70)')
plt.axhline(30,color = 'green', linestyle = 'dotted', label = 'Oversold (30)')
plt.title('RSI 14 Over Time')
plt.title(ticker, fontsize = 10, y=1, loc='right')
plt.xlabel('Date')
plt.ylabel('RSI Value')
plt.legend()
plt.grid(True)
plt.show()

#Volume 
plt.figure(figsize=(14,4))
plt.plot(df["Volume"], color ='orange')
plt.title('Trading Volume')
plt.title(ticker, fontsize = 10, y=1, loc='right')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.show()

#Prediction vs Actual 
plt.figure(figsize=(10, 5))
plt.plot(Y_test.values[:50], label='Actual', marker='o')
plt.plot(predictions[:50], label='Predicted', marker='x')
plt.title('Predicted vs Actual (Sample)')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()
plt.grid(True)
plt.show()


# Save the best model
joblib.dump(best_rf, "apple_rf_model.pkl")
