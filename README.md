# trabalho_de_pad
Trabalho da disciplina Pensamento Analítico de Dados - Bacharelado Inteligência Artificial

**Integrantes**: Cibelle Pires Botelho, Yasmin Mageste, Julia Jaime Biancolini, Giovana Faleiro, Gustavo Pires Fontana e Felipe Rodrigues
**tema**: Análise de sentimentos

# Projeto FMF - Análise de sentimentos
referência: <https://artificialcorner.com/p/first-ml-model>

## Contexto

A análise de sentimentos é uma das tarefas mais recorrentes em Processamento de Linguagem Natural (PLN), com aplicações em sistemas de recomendação, monitoramento de reputação de marcas, análise de avaliações de produtos e pesquisas de opinião.

O projeto **FMF — First Machine Learning Model** utiliza esse domínio como contexto para apresentar, de forma didática, as principais etapas de construção de um pipeline de Machine Learning supervisionado para classificação de textos.

## Problema Abordado

O problema tratado é uma tarefa de **classificação binária de texto**. A partir do conteúdo de uma crítica cinematográfica, o modelo deve classificá-la como:

* `positive`: avaliação positiva;
* `negative`: avaliação negativa.

Trata-se de um problema de **aprendizado supervisionado**, pois cada crítica presente no conjunto de dados possui um rótulo conhecido, utilizado durante o treinamento dos modelos.

## Dataset

O projeto utiliza o **IMDB Dataset**, disponibilizado por meio da plataforma Kaggle. O conjunto original contém aproximadamente **50.000 críticas de filmes**, divididas entre avaliações positivas e negativas.

As principais variáveis são:

* `review`: texto completo da crítica cinematográfica;
* `sentiment`: sentimento associado à crítica, classificado como positivo ou negativo.

Para fins didáticos, o projeto utiliza uma amostra de 10.000 registros, inicialmente composta por:

* 9.000 avaliações positivas;
* 1.000 avaliações negativas.

Essa configuração introduz propositalmente um cenário de desbalanceamento de classes.

## Tecnologias e Bibliotecas Utilizadas

O projeto utiliza as seguintes bibliotecas:

* `pandas`: leitura, manipulação, seleção e amostragem dos dados;
* `imbalanced-learn`: tratamento do desbalanceamento das classes por meio de undersampling;
* `scikit-learn`: divisão dos dados, vetorização dos textos, treinamento dos modelos, avaliação de desempenho e otimização de hiperparâmetros.

## Estrutura Geral do Pipeline

O projeto segue um pipeline composto pelas seguintes etapas:

1. **Aquisição dos dados:** obtenção do dataset de críticas do IMDb por meio do Kaggle.

2. **Amostragem dos dados:** seleção de uma amostra de 10.000 avaliações para execução do experimento.

3. **Balanceamento das classes:** aplicação de undersampling para reduzir a quantidade de registros da classe majoritária e produzir um conjunto balanceado.

4. **Divisão entre treino e teste:** separação dos dados em conjuntos de treinamento e avaliação.

5. **Vetorização textual:** transformação das críticas em representações numéricas utilizando TF-IDF.

6. **Modelagem:** treinamento e comparação de diferentes algoritmos clássicos de classificação.

7. **Avaliação:** análise dos resultados por meio de métricas de classificação.

8. **Otimização:** ajuste dos hiperparâmetros do melhor modelo com `GridSearchCV`.

## Preparação e Balanceamento dos Dados

O projeto cria inicialmente um conjunto desbalanceado com 9.000 avaliações positivas e 1.000 avaliações negativas.

Para evitar que os classificadores favoreçam a classe majoritária, é utilizado o `RandomUnderSampler`. Essa técnica reduz aleatoriamente a quantidade de exemplos da classe mais frequente até que ambas as classes possuam a mesma quantidade de registros.

Após o balanceamento, o conjunto utilizado contém:

* 1.000 avaliações positivas;
* 1.000 avaliações negativas.

Embora o undersampling facilite o treinamento e a avaliação equilibrada das classes, ele também descarta parte dos dados disponíveis.

## Divisão entre Treinamento e Teste

Os dados balanceados são separados em:

* 67% para treinamento;
* 33% para teste.

O conjunto de treinamento é utilizado para ajustar os parâmetros dos modelos. O conjunto de teste é reservado para avaliar o desempenho dos classificadores em dados que não foram utilizados durante o aprendizado.

## Vetorização dos Textos

Como os algoritmos tradicionais de Machine Learning não processam textos diretamente, as críticas precisam ser convertidas em representações numéricas.

O projeto apresenta duas abordagens:

* `CountVectorizer`: representa cada documento pela frequência de ocorrência das palavras;
* `TfidfVectorizer`: pondera as palavras considerando sua frequência no documento e sua relevância em relação ao restante do corpus.

A técnica escolhida é o **TF-IDF**, pois reduz a importância de palavras muito frequentes e atribui maior peso aos termos mais representativos de cada crítica.

O vetorizador é ajustado exclusivamente no conjunto de treinamento. No conjunto de teste, utiliza-se apenas a transformação previamente aprendida, evitando vazamento de dados.

## Modelos Avaliados

O projeto compara quatro algoritmos de classificação:

* Support Vector Machine — SVM;
* Regressão Logística;
* Árvore de Decisão;
* Naive Bayes.

Os resultados aproximados foram:

| Modelo              | Acurácia |
| ------------------- | -------: |
| SVM                 |      84% |
| Regressão Logística |      83% |
| Árvore de Decisão   |      64% |
| Naive Bayes         |      63% |

O melhor desempenho foi obtido pelo **SVM com kernel linear**.

## Métricas de Avaliação

O desempenho dos modelos é analisado por meio das seguintes métricas:

* **Acurácia:** proporção total de previsões corretas;
* **Precisão:** proporção de previsões corretas entre os exemplos classificados em determinada classe;
* **Recall:** proporção de exemplos reais de uma classe identificados corretamente;
* **F1-score:** média harmônica entre precisão e recall;
* **Relatório de classificação:** apresenta precision, recall, F1-score e suporte para cada classe;
* **Matriz de confusão:** mostra a distribuição dos acertos e erros entre as classes.

O modelo SVM apresentou aproximadamente 84% de acurácia, com F1-scores equilibrados entre as classes positiva e negativa, próximos de 0,84 e 0,83.

## Otimização de Hiperparâmetros

Após a comparação inicial, o projeto utiliza o `GridSearchCV` para buscar a melhor configuração do SVM.

São testados diferentes valores para:

* parâmetro de regularização `C`;
* tipo de kernel;
* combinações avaliadas por validação cruzada.

A melhor configuração encontrada foi:

```python
SVC(C=1, kernel="linear")
```

O `GridSearchCV` permite comparar sistematicamente diferentes combinações de hiperparâmetros, evitando que o ajuste seja realizado apenas por tentativa e erro manual.

## Características do Projeto

Entre as principais características do projeto estão:

* aplicação prática de classificação de sentimentos;
* introdução proposital de um cenário de desbalanceamento;
* balanceamento das classes com undersampling;
* comparação entre `CountVectorizer` e TF-IDF;
* treinamento de quatro algoritmos clássicos de Machine Learning;
* avaliação com diferentes métricas de classificação;
* comparação objetiva do desempenho dos modelos;
* otimização do melhor classificador com `GridSearchCV`;
* separação adequada entre ajuste do vetorizador e transformação dos dados de teste.

## Limitações Observadas

O projeto apresenta algumas limitações:

* a amostra de 10.000 registros representa apenas 20% do conjunto original de 50.000 críticas;
* após o undersampling, somente 2.000 registros são efetivamente utilizados;
* o descarte de exemplos da classe majoritária pode eliminar informações relevantes;
* não são avaliadas técnicas de oversampling ou geração de dados sintéticos;
* o pré-processamento textual é relativamente simples;
* não são discutidos procedimentos como remoção de stopwords, lematização ou tratamento de negações;
* a avaliação utiliza apenas uma divisão entre treino e teste para a comparação inicial;
* apenas modelos clássicos de Machine Learning são analisados;
* abordagens baseadas em embeddings, como Word2Vec, GloVe e FastText, não são consideradas;
* modelos baseados em Transformers, como BERT, RoBERTa e DistilBERT, não são avaliados;
* os resultados podem variar de acordo com a amostragem e com o estado aleatório utilizado;
* não há análise detalhada dos erros de classificação produzidos pelo modelo.


## Conclusão

O projeto apresenta de forma didática as principais etapas de um pipeline clássico de Machine Learning aplicado à classificação de sentimentos.

O experimento demonstra que a preparação dos dados, o balanceamento das classes, a vetorização dos textos e a escolha do algoritmo influenciam diretamente o desempenho final. Entre os modelos avaliados, o SVM com kernel linear e `C=1` apresentou o melhor resultado, alcançando aproximadamente 84% de acurácia.

Apesar das limitações, o projeto oferece uma base adequada para compreender o funcionamento de tarefas supervisionadas de classificação textual e pode ser expandido posteriormente com técnicas mais avançadas de PLN e Deep Learning.


## A - ASK 
Levando em consideração o contexto de revisões de filmes, conseguimos dizer se a revisão foi positiva ou negativa com base no que está escrito?
