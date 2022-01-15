import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import joblib

df = sns.load_dataset('penguins')
df = df.dropna(subset=['sex'])
df['sex_int'] = df['sex'].map({'Male': 0, 'Female': 1})

one_hot = OneHotEncoder()
encoded = one_hot.fit_transform(df[['island']])
df[one_hot.categories_[0]] = encoded.toarray()
df = df.drop(columns=['island', 'sex'])

X = df.iloc[:, 1:]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=100)

forest = RandomForestClassifier(n_estimators=100, random_state=100)

forest.fit(X_train, y_train)
predictions = forest.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, predictions))

print(X_test.iloc[0])

print("Saving model...")
joblib.dump(forest, "penguins.joblib")
print("Model saved!")


