# %%
# imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = pd.read_csv('data//emails.csv')

# %% 
data.dropna(axis=1)
data = data.drop_duplicates(keep='first')
data.sample(4)

# %% normalize word frequencies
word_freq = data.iloc[:,1:3001]
scaler = MinMaxScaler()
word_freq_normalized = scaler.fit_transform(word_freq)

df_word_freq_normalized = pd.DataFrame(word_freq_normalized, columns=word_freq.columns)
data.iloc[:, 1:3001] = df_word_freq_normalized

# %% split dataset
X= data.iloc[:, 1:3001]
y = data.iloc[:, -1]

# %%
plt.pie(y.value_counts(), labels=['Not Spam', 'Spam'], autopct="%0.2f")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# %%
# Initialize the models
log_reg = LogisticRegression(max_iter=1000)
naive_bayes = MultinomialNB()
decision_tree = DecisionTreeClassifier()

# Train the models
log_reg.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)

# Predict on the test set
log_reg_pred = log_reg.predict(X_test)
naive_bayes_pred = naive_bayes.predict(X_test)
decision_tree_pred = decision_tree.predict(X_test)

# %%
def evaluate_model(name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    print(f"Performance of {name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("-" * 30)

# Evaluate all models
evaluate_model("Logistic Regression", y_test, log_reg_pred)
evaluate_model("Naive Bayes", y_test, naive_bayes_pred)
evaluate_model("Decision Tree", y_test, decision_tree_pred)

# %%
