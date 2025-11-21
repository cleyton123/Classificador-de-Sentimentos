import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Carregar dataset
df = pd.read_csv("data/reviews.csv")

X = df["text"]
y = df["label"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Vetorização TF-IDF
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Avaliação
predictions = model.predict(X_test_vect)
print("Acurácia:", accuracy_score(y_test, predictions))
print("\nRelatório:\n", classification_report(y_test, predictions))

# Salvar modelo
with open("model/sentiment_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("\nModelo treinado e salvo!")
