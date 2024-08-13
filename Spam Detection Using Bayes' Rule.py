import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
spam_df = pd.read_csv('emails.csv')
ham = spam_df[spam_df['spam'] == 0]
spam = spam_df[spam_df['spam'] == 1]
spam_percentage = (len(spam) / len(spam_df)) * 100
ham_percentage = (len(ham) / len(spam_df)) * 100
print(f'Spam Percentage = {spam_percentage:.2f}%')
print(f'Ham Percentage = {ham_percentage:.2f}%')
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(spam_df['text'])
y = spam_df['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
y_predict_test = NB_classifier.predict(X_test)
mse = mean_squared_error(y_test, y_predict_test)
r_squared = r2_score(y_test, y_predict_test)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R-squared: {r_squared:.4f}')
cm = confusion_matrix(y_test, y_predict_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.show()
