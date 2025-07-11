Exercício Prático 4.1
1. Objetivos
Este exercício tem como objetivo integrar testes automatizados com o GitHub Actions, criando um pipeline de CI/CD (Integração Contínua/Entrega Contínua) para um projeto de Machine Learning. O objetivo é garantir que o código, as funções e o modelo funcionem corretamente sempre que haja alterações no repositório. Mais especificamente:

Criar funções e testes com pytest;
Automatizar a execução desses testes com o GitHub Actions;
Treinar um modelo de Machine Learning real e testar sua integridade;
2 Introdução
2.1 O que é CI/CD?
CI/CD (Integração Contínua / Entrega Contínua) é uma prática de DevOps que permite automatizar o processo de teste e entrega de software. A ideia é garantir que, a cada alteração no código (CI), o projeto seja testado automaticamente e, se estiver tudo certo, possa ser entregue rapidamente (CD).

CI (Integração Contínua): Testes automatizados são rodados a cada solicitação push ou pull.

CD (Entrega Contínua): Código aprovado pode ser implantado automaticamente em produção ou em ambientes de preparação.

2.2 E o que é GitHub Actions ?
GitHub Actions é uma plataforma de integração contínua e entrega contínua (CI/CD) do GitHub. Ela permite automatizar fluxos de trabalho diretamente a partir do repositório, sem depender de ferramentas externas.

Foi lançado oficialmente em 2019 pela empresa GitHub, que hoje pertence à Microsoft. Com ele, os desenvolvedores podem definir tarefas que devem acontecer sempre que algo ocorra no repositório — por exemplo, ao fazer um push, abrir um pull request ou criar uma nova tag. Essas tarefas automatizadas são chamadas de "ações" e ficam definidas em arquivos YAML;

Alguns casos de uso comuns são: testar código automaticamente sempre que ele for alterado (ex: rodar pytest, unittest, lint), treinar e validar modelos de machine learning sempre que o código do pipeline for modificado, deploy automático de aplicações (como APIs em FastAPI ou modelos em produção), atualização de documentação sempre que arquivos README.md ou docs/forem modificados e análise de segurança do código-fonte.

O GitHub Actions é nativamente integrado à interface do GitHub. Isso significa que cada vez que um fluxo de trabalho for executado, você poderá ver o progresso diretamente no site, com registros detalhados de cada passo. É possível visualizar fluxos de trabalho passando, falhando, sendo cancelados, etc. Também é possível baixar artefatos gerados durante os testes (como modelos salvos, relatórios, etc.)

Para saber mais, recomendamos (fortemente) dar uma olhada na página inicial do Github Actions

3. Criando a Primeira Ação/Pipeline
Agora que entendemos o que é CI/CD e como o GitHub Actions funciona, vamos colocar a mão na massa. Nesta etapa, vamos criar nossa primeira ação para automatizar a execução de testes Python sempre que fizermos uma solicitação push ou pull para o repositório. A ideia aqui é simular o uso de um pipeline simples de integração contínua, que verifica automaticamente se nosso código está funcionando corretamente.

Crie um repositório no github, clone e entre nele;

Vamos criar um código simples para ser testado: algumas funções matemáticas. Crie a pasta funcoes e dentro dela crie dois arquivos: um vazio chamado __init__.py(que serve para o python entender o código como um pacote), e outro chamado matematica.py, com o seguinte conteúdo:

def soma(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("Não pode dividir por zero!")
    return a / b

# Crie outras funções
Agora crie uma pasta de testes e dentro dela crie novamente um arquivo vazio __init__.pye o seguinte arquivo test_matematica.py:
from funcoes import matematica

def test_soma():
    assert matematica.soma(2, 3) == 5

def test_divide():
    assert matematica.divide(10, 2) == 5

def test_divide_zero():
    try:
        matematica.divide(10, 0)
        assert False
    except ValueError:
        assert True

# Crie testes para todos os casos das outras funções que você criou
Vamos executar esses testes a partir de uma biblioteca chamada pytest , que é uma biblioteca muito utilizada para criar testes unitários (testes que testam um único comportamento) e funcionais (que testam uma funcionalidade maior) em python. Instale com pip ( pip install pytest ) e execute o comando pytest, que executa todos os testes da massa testes que começam com "test_". É esse o comando que executaremos nas ações.

Você pode testar na mão os testes, basta rodar pytest. Teste quebrando um teste, para ver a interface.

Crie um arquivo requirements.txtna raiz do projeto, e coloque a lib pytest nela (criamos esse arquivo pro github saber quais libs tem que instalar, já que não sabemos o que tem no nosso ambiente).

Crie uma pasta chamada .github/workflowsna raiz do seu projeto. Essa é a estrutura padrão esperada pelo GitHub Actions para encontrar seus fluxos de trabalho, então não modifique o nome (não esqueça do ponto no começo).

Dentro dessa pasta, crie um arquivo chamado ci_python_tests.yml . Esse arquivo define um fluxo de trabalho que será disparado automaticamente quando houver atualizações no repositório.

Adicione o seguinte conteúdo ao arquivo:

name: Testes com Pytest

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configurar Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Instalar dependências
        run: |
          pip install -r requirements.txt
      - name: Rodar testes
        run: pytest
Esse arquivo descreve um pipeline com as seguintes etapas:

Ele será executado em qualquer push para o principal da filial;
Ele configura/define o Python;
Instalar as dependências do projeto (como pytest);
E por fim, execute os testes que estiverem no projeto.
Com tudo pronto, suba todas essas mudanças para o repositório (use esta mensagem):

git add .
git commit -m "Criando o primeiro pipeline"
git push
Entre no link do seu repositório e vá na aba actions, lá você vai conseguir acompanhar a ação que você criou sendo realizada.

4. Criando um Pipeline para Machine Learning
Na etapa anterior, criamos um fluxo de trabalho de testes com pytest para validar funções simples. Agora, vamos simular um caso mais próximo de um cenário real de MLOps.

Crie um arquivo chamado pipeline.py, onde obterá o código do pipeline que você desenvolveu no módulo 2. Vamos utilizá-lo em nossos testes. Coloque na pasta funções (ou crie uma com nome mais protegido).
Se por acaso não criou um pipeline, você pode usar o básico abaixo:
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def create_pipeline(X):
    numerical_features = X.select_dtypes(include='number').columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    return model_pipeline
Crie um arquivo test_pipeline.pydentro de testes , onde você escreverá os testes unitários para o modelo. Adicione o seguinte código:
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split

from pipeline import create_pipeline

df = pd.read_csv('adult.csv', na_values=['#NAME?']) # Ajuste o caminho conforme necessário
X = df.drop('income', axis=1)
y = df['income']

def test_pipeline_fit():
    pipe = pipeline.create_pipeline(X)  # Só passa X agora
    pipe.fit(X, y)
    check_is_fitted(pipe)

def test_pipeline_predict_shape():
    pipe = pipeline.create_pipeline(X)
    pipe.fit(X.iloc[:20], y.iloc[:20])
    y_pred = pipe.predict(X.iloc[:20])
    assert len(y_pred) == 20
Os testes fazem o seguinte:

o primeiro verifica se o modelo foi treinado
o segundo checa se o modelo faz as predições como esperado, comparando se a quantidade de predições é a mesma da quantidade de entradas (20)
Adicione uma cópia do nosso conjunto de dados ( adult.csv ) dentro da raiz do projeto (ou, se quiser mais organizado, crie uma pasta e passe o caminho corretamente na função pd.read_csv() ).

Atualize o requirements.txt com as bibliotecas permitidas para executar os dois novos códigos.

Suba o novo código para o github, e (se tudo certo) o pipeline de CI/CD também vai testar o modelo!

git add .
git commit -m "Testando o modelo"
git push
5. Melhorando o Pipeline para Machine Learning
Os testes que criamos até o momento são bem simples (como testes unitários devem ser), mas muito mais que podem ser feitos num pipeline para ML, até para além dos modelos. Vamos criar alguns testes mais interessantes:

Teste de integridade dos dados: verifique se o conjunto de treino/teste não possui valores nulos ou tipos inesperados.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

def test_no_nulls_in_data():
    assert not X_train.isnull().any().any(), "X_train possui valores nulos"
    assert not X_test.isnull().any().any(), "X_test possui valores nulos"
    assert not y_train.isnull().any().any(), "y_train possui valores nulos"
    assert not y_test.isnull().any().any(), "y_test possui valores nulos"
Teste de desempenho mínimo: garanta que o código só será aceito se o modelo atingir um nível aceitável de desempenho.
def test_model_performance():
    pipe = pipeline.create_pipeline(X_train)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.75, f"Acurácia abaixo do esperado: {acc:.2f}"
Teste funcional do pipeline: verifique se o pipeline pode ser treinado e preveja sem erro em um subconjunto limitado de dados (amostra)
def test_pipeline_runs_on_sample():
    pipe = pipeline.create_pipeline(X_train)
    X_sample = X_train.sample(10, random_state=42)
    y_sample = y_train.loc[X_sample.index]

    try:
        pipe.fit(X_sample, y_sample)
        preds = pipe.predict(X_sample)
    except Exception as e:
        assert False, f"Pipeline falhou ao rodar em amostra: {e}"
Teste de reprodutibilidade simples: verifica se duas execuções seguidas do modelo resultam em predições idênticas
def test_reproducibility():
    pipe1 = pipeline.create_pipeline(X_train)
    pipe2 = pipeline.create_pipeline(X_train)

    pipe1.fit(X_train, y_train)
    pipe2.fit(X_train, y_train)

    pred1 = pipe1.predict(X_test)
    pred2 = pipe2.predict(X_test)

    assert np.array_equal(pred1, pred2), "Predições diferentes entre execuções"
Teste de colunas esperadas: garantir que colunas esperadas estejam presentes (importante em produção)
def test_expected_columns():
    expected_cols = ['education', 'age', 'fnlwgt']
    for col in expected_cols:
        assert col in X_train.columns, f"Coluna ausente no dataset: {col}"
Tente criar mais alguns testes que você acha (pode pesquisar na internet por testes comuns relevantes de modelos). Ao final, coloque a mensagem de commit "Melhorando o pipeline"

6. Usando o picles
Existe uma biblioteca muito legal e bastante utilizada que ainda não usamos nesse curso, que é o pickle . Ela é usada para serializar e desserializar objetos, ou seja, transformar objetos Python ou modelos de ML em um formato binário que pode ser salvo em um arquivo e depois carregado novamente, com todos os dados preservados. Em MLOps, é bastante útil permitir salvar um modelo já treinado, com todos os pesos e parâmetros, junto com todo o pipeline de processamento, em um único arquivo! Isso facilita muito a reutilização, evitando re-treinamentos (que depender do modelo, é bem caro).

Perceba que em nossos testes estamos sempre treinando um modelo do zero, o que nem sempre é o ideal: podemos ter um ou mais testes que testamos o treino em si, mas nos outros isso não é necessário. Dessa forma, realize as seguintes tarefas:

Estude como colocar todo o nosso pipeline num arquivo .pkl, e depois como ler e carregar o modelo.
Atualizar os testes onde for possível usar o modelo serializado
Crie novos que testem essa serialização/desserialização.
Suba o código, para disparar a ação, com a seguinte mensagem:
git add .
git commit -m "Adicionando pickle"
git push