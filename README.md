# trabalho_de_pad
Trabalho da disciplina Pensamento Analítico de Dados - Bacharelado Inteligência Artificial

**Integrantes**: Cibelle Pires Botelho, Yasmin Mageste, Julia Jaime Biancolini, Giovana Faleiro, Gustavo Pires Fontana e Felipe Rodrigues
**tema**: Análise de sentimentos

# Projeto FMF - Análise de sentimentos
referência: <https://artificialcorner.com/p/first-ml-model>

## Contexto

A análise de sentimentos é uma das tarefas mais recorrentes em Processamento de Linguagem Natural (PLN), com aplicações em sistemas de recomendação, monitoramento de reputação de marcas, análise de avaliações de produtos e pesquisas de opinião.

O projeto utiliza esse domínio como contexto para apresentar, de forma didática, as principais etapas de construção de um pipeline de Machine Learning supervisionado para classificação de textos.

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

# Etapas do AGEMC
## ASK — Formulação do Problema
A etapa inicial do projeto FMF parte da pergunta: **“Levando em consideração o contexto de revisões de filmes, conseguimos dizer se a revisão foi positiva ou negativa com base no que está escrito?”** Com base nessa questão, o objetivo é desenvolver um modelo capaz de interpretar o conteúdo textual de uma crítica e atribuir a ela uma das duas categorias possíveis de sentimento. Dessa forma, o projeto é estruturado como um problema de **classificação binária de texto**, no qual a crítica representa a informação de entrada e o sentimento, positivo ou negativo, representa a resposta esperada. Como cada exemplo do conjunto de dados já possui uma classificação associada, o aprendizado ocorre de maneira supervisionada, permitindo que o modelo identifique padrões presentes na linguagem e os utilize na análise de novas revisões.

## GET — Obtenção dos Dados

Nesta etapa, são reunidos os dados necessários para o desenvolvimento do projeto. A base utilizada é o **IMDB Dataset**, disponibilizado publicamente na plataforma Kaggle e composto por 50.000 avaliações de filmes. O conjunto apresenta duas variáveis principais: `review`, que armazena o texto escrito pelo usuário, e `sentiment`, que indica se a avaliação foi classificada como positiva ou negativa. Para tornar o experimento mais simples e reduzir o tempo de processamento, é selecionada uma amostra de 10.000 registros. Essa amostra é construída de forma propositalmente desbalanceada, contendo 9.000 críticas positivas e 1.000 negativas, permitindo que o projeto também explore os impactos do desequilíbrio entre classes e as estratégias utilizadas para corrigi-lo nas etapas posteriores.

link dos dados: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## EXPLORE — Exploração e Preparação dos Dados

A etapa de exploração tem como objetivo compreender a estrutura do conjunto de dados e prepará-lo para o treinamento dos modelos. Nesse momento, são observadas a distribuição das classes, a quantidade de registros disponíveis, as variáveis presentes e a forma como os textos serão representados numericamente. Essa análise é importante porque problemas como desbalanceamento, divisão inadequada dos dados ou vazamento de informações podem comprometer a avaliação dos classificadores.

### a) Análise e Tratamento do Desbalanceamento

A amostra inicial contém 10.000 críticas, distribuídas de forma desigual entre as classes: 9.000 avaliações positivas e 1.000 negativas. Esse desbalanceamento pode levar o modelo a favorecer a classe majoritária, alcançando uma acurácia aparentemente elevada mesmo sem aprender adequadamente os padrões associados às críticas negativas.

Para corrigir esse problema, é aplicada a técnica de **undersampling aleatório** com a classe `RandomUnderSampler`, da biblioteca `imbalanced-learn`. O procedimento reduz a quantidade de exemplos positivos até igualá-la à quantidade de exemplos negativos. Ao final, é obtido o conjunto balanceado `df_review_bal`, composto por 2.000 registros, sendo 1.000 avaliações positivas e 1.000 negativas.

O balanceamento permite que as duas classes tenham a mesma influência durante o treinamento. Entretanto, como o undersampling descarta parte dos registros da classe majoritária, essa solução também reduz a quantidade de informações disponíveis para o modelo.

### b) Separação entre Treinamento e Teste

Após o balanceamento, os dados são divididos em conjuntos de treinamento e teste por meio da função `train_test_split`, disponibilizada pelo `scikit-learn`. A divisão reserva 67% dos registros para treinamento e 33% para teste.

O conjunto de treinamento é utilizado para que os algoritmos identifiquem padrões linguísticos associados aos sentimentos positivo e negativo. Já o conjunto de teste permanece separado durante o aprendizado e é utilizado somente para verificar o comportamento dos modelos diante de críticas não observadas anteriormente.

Considerando o conjunto balanceado de 2.000 registros, a divisão resulta em aproximadamente 1.340 críticas para treinamento e 660 para teste. Para preservar a proporção das classes em ambas as partes, a divisão pode ser realizada de forma estratificada, utilizando o parâmetro `stratify`.

### c) Conversão dos Textos em Representações Numéricas

Os textos das críticas não podem ser processados diretamente pelos algoritmos clássicos de Machine Learning. Por isso, eles precisam ser transformados em vetores numéricos que representem as palavras presentes em cada documento.

O projeto apresenta duas técnicas baseadas no modelo **Bag of Words**, em que cada palavra do vocabulário corresponde a uma característica:

* `CountVectorizer`: representa cada crítica pela quantidade de vezes que cada palavra aparece. Nessa abordagem, termos com a mesma frequência recebem o mesmo valor, independentemente de sua relevância no restante do conjunto;
* `TfidfVectorizer`: calcula o peso de cada palavra considerando sua frequência dentro da crítica e sua raridade no corpus. Palavras muito comuns em vários textos recebem menor peso, enquanto termos mais específicos e representativos recebem maior importância.

O projeto escolhe o **TF-IDF**, pois a classificação de sentimentos depende da identificação de termos capazes de diferenciar avaliações positivas e negativas. Palavras como “excellent”, “wonderful”, “boring” ou “terrible” tendem a apresentar maior valor discriminativo do que palavras frequentes e pouco informativas.

### d) Prevenção de Vazamento de Dados

O vetorizador TF-IDF deve aprender o vocabulário e os pesos exclusivamente a partir do conjunto de treinamento. Por esse motivo, utiliza-se `fit_transform()` nos textos de treino e apenas `transform()` nos textos de teste.

Esse procedimento impede que palavras, frequências ou padrões presentes no conjunto de teste influenciem a representação utilizada durante o treinamento. Caso o vetorizador fosse ajustado antes da divisão dos dados, ocorreria vazamento de informação, tornando as métricas finais excessivamente otimistas.

### e) Matriz de Características

Após a vetorização, cada linha da matriz representa uma crítica e cada coluna corresponde a uma palavra do vocabulário identificado no conjunto de treinamento. No experimento, a matriz de treinamento possui aproximadamente **1.340 críticas por 20.625 termos**.

Como cada crítica contém apenas uma pequena parcela do vocabulário total, a maior parte dos valores da matriz é igual a zero. Por isso, o `TfidfVectorizer` utiliza uma representação esparsa, que armazena apenas os valores diferentes de zero e reduz o consumo de memória.

Ao final desta etapa, os dados encontram-se balanceados, separados corretamente entre treinamento e teste e representados em formato numérico, ficando prontos para a etapa de modelagem.

## MODEL — Seleção e Treinamento dos Modelos

Nesta etapa, os dados já balanceados e representados numericamente por meio do TF-IDF são utilizados para treinar diferentes algoritmos de classificação supervisionada. O projeto compara quatro modelos: **Support Vector Machine (SVM)**, **Regressão Logística**, **Árvore de Decisão** e **Naive Bayes**. Todos recebem como entrada a matriz TF-IDF do conjunto de treinamento e os respectivos rótulos de sentimento, garantindo uma comparação sob as mesmas condições. Nos resultados iniciais, o SVM alcança aproximadamente 84% de acurácia, seguido pela Regressão Logística, com 83%, enquanto a Árvore de Decisão e o Naive Bayes obtêm cerca de 64% e 63%, respectivamente. O melhor desempenho dos modelos lineares está relacionado à natureza dos dados textuais vetorizados, que geram uma matriz esparsa, de alta dimensionalidade e com grande quantidade de características. Após essa comparação, o SVM é selecionado como modelo principal do projeto e passa por uma etapa de otimização com `GridSearchCV`. Esse procedimento testa diferentes combinações de hiperparâmetros, como os valores do parâmetro de regularização `C` e os kernels `linear` e `rbf`, utilizando validação cruzada com cinco divisões. Em cada combinação, o conjunto de treinamento é repartido em cinco partes, sendo quatro utilizadas para treinar o modelo e uma para validação, alternando-se a parte de validação até que todas tenham sido avaliadas. Ao final da busca, a configuração com melhor desempenho médio é o **SVM com kernel linear e `C=1`**, escolhido para a avaliação final no conjunto de teste.

## COMMUNICATE — Avaliação e Comunicação dos Resultados

Nesta etapa, os resultados obtidos pelos modelos são interpretados e apresentados de forma a responder à pergunta definida no início do projeto. A comparação das métricas mostra que o **Support Vector Machine (SVM)** foi o classificador com melhor desempenho, alcançando aproximadamente **84% de acurácia**. O modelo também apresentou resultados equilibrados entre as duas classes, com F1-score próximo de **0,84 para críticas positivas** e **0,83 para críticas negativas**, indicando que ele consegue reconhecer ambos os sentimentos sem favorecer excessivamente uma categoria. O relatório de classificação complementa essa análise ao detalhar precisão, revocação e F1-score para cada classe. Já a matriz de confusão permite observar os acertos e erros de forma mais específica: foram registradas 290 classificações corretas de uma classe e 265 da outra, além de 45 falsos positivos e 60 falsos negativos. Esses resultados comunicam que os padrões extraídos pelo TF-IDF são suficientemente informativos para distinguir grande parte das avaliações positivas e negativas, embora ainda existam limitações em textos ambíguos, irônicos ou que contenham sentimentos mistos. De forma geral, o experimento demonstra que a combinação entre vetorização TF-IDF e SVM linear constitui uma solução consistente para a classificação automática de sentimentos no conjunto de críticas do IMDb.

### Objetivo científico do projeto

* Investigar se o conteúdo textual de uma crítica cinematográfica contém informações suficientes para identificar automaticamente o sentimento expresso pelo usuário.
* Avaliar a aplicação de algoritmos clássicos de Machine Learning supervisionado em uma tarefa de classificação binária de textos.
* Comparar diferentes classificadores sob as mesmas condições de treinamento e representação dos dados.
* Verificar quais modelos apresentam maior capacidade de generalização para críticas não utilizadas durante o treinamento.
* Analisar se a representação TF-IDF consegue capturar palavras e padrões relevantes para separar avaliações positivas e negativas.

### O que o projeto esperava obter desde o início?

* Construir um modelo capaz de aprender associações entre palavras presentes nas críticas e seus respectivos rótulos de sentimento.
* Identificar qual dos algoritmos avaliados apresenta o melhor desempenho para o problema.
* Obter resultados equilibrados para as classes positiva e negativa, evitando que o modelo favoreça apenas uma delas.
* Demonstrar, de forma didática, as etapas necessárias para desenvolver um pipeline completo de classificação textual.
* Produzir evidências quantitativas de que o sentimento de uma crítica pode ser estimado a partir de seu conteúdo escrito.

### O que o projeto deseja prever, estimar ou permitir?

* **Prever** se uma nova crítica de filme expressa um sentimento positivo ou negativo.
* **Estimar** a capacidade de diferentes algoritmos de reconhecer padrões linguísticos associados a cada sentimento.
* **Permitir** a classificação automática de grandes volumes de avaliações textuais.
* **Auxiliar** processos de análise de opinião, monitoramento de satisfação e organização de feedbacks.
* **Servir como base** para aplicações futuras mais avançadas, utilizando embeddings, redes neurais ou modelos baseados em Transformers.

## Próximos Passos

### Semana 3 — Modelagem e Comunicação

Estudo da implementação dos modelos utilizados no projeto de referência, incluindo a análise do processo de treinamento, comparação de desempenho e avaliação das métricas. Também será aprofundada a etapa de comunicação dos resultados, buscando compreender como as conclusões do projeto são apresentadas e relacionadas ao problema inicialmente definido.

### Semana 4 — Definição e Implementação da Originalidade

Definição da contribuição original do projeto, identificando qual etapa do ciclo AGEMC será modificada, ampliada ou adaptada. Nesta semana, também será realizada a implementação da proposta escolhida, com a descrição das alterações realizadas e da justificativa para sua aplicação.

### Semana 5 — Apresentação Final

Preparação e realização de uma apresentação para a turma, reunindo todas as etapas desenvolvidas ao longo do projeto. A apresentação deverá abordar o problema investigado, a obtenção e exploração dos dados, a modelagem, a comunicação dos resultados e a contribuição original implementada.

