import pickle

# Carregar modelo
with open("model/sentiment_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Função para classificar
def classificar_sentimento(texto):
    vetor = vectorizer.transform([texto])
    pred = model.predict(vetor)[0]
    return pred

# Teste
if __name__ == "__main__":
    texto = input("Digite um texto para analisar: ")
    resultado = classificar_sentimento(texto)
    print("Sentimento:", resultado)
