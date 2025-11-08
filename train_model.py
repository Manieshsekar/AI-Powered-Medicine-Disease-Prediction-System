import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('disease_symptom_small.csv')

# Get the list of symptom columns (everything except 'diseases')
symptom_cols = list(df.columns)
symptom_cols.remove('diseases')

X = df[symptom_cols].values
y = df['diseases'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = MultinomialNB()
clf.fit(X_train, y_train)
print("Accuracy:", clf.score(X_test, y_test))

pickle.dump(clf, open('model.pkl', 'wb'))
# Save the columns used
with open('symptom_columns.pkl', 'wb') as f:
    pickle.dump(symptom_cols, f)
