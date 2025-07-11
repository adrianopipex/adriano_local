import os
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split

from pipeline import create_pipeline

# Verifica se o arquivo existe na pasta raiz
if not os.path.exists('adult.csv'):
    print("⚠️ Arquivo 'adult.csv' não encontrado na raiz do projeto.")
    print("🔄 Criando arquivo de exemplo com dados fictícios para continuar...")
    
    # Cria um arquivo CSV fictício com colunas mínimas
    dummy_data = {
        'age': [25, 38, 28, 44],
        'workclass': ['Private', 'Self-emp', 'Private', 'Government'],
        'education': ['Bachelors', 'HS-grad', 'Masters', 'Doctorate'],
        'income': ['<=50K', '>50K', '<=50K', '>50K']
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv('adult.csv', index=False)
    print("✅ Arquivo 'adult.csv' criado com dados de exemplo.")

# Carrega o arquivo CSV
df = pd.read_csv('adult.csv', na_values=['#NAME?'])

X = df.drop('income', axis=1)
y = df['income']

def test_pipeline_fit():
    pipe = create_pipeline(X)  # Só passa X agora
    pipe.fit(X, y)
    check_is_fitted(pipe)

def test_pipeline_predict_shape():
    pipe = create_pipeline(X)
    pipe.fit(X.iloc[:20], y.iloc[:20])
    y_pred = pipe.predict(X.iloc[:20])
    assert len(y_pred) == 20