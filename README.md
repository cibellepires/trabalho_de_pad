# trabalho_de_pad
Trabalho da disciplina Pensamento Analítico de Dados - Bacharelado Inteligência Artificial

**Integrantes**: Cibelle Pires Botelho, Yasmin Mageste, Julia Jaime Biancolini, Giovana Faleiro, Gustavo Pires Fontana e Felipe Rodrigues<br>
**Tema**: Análise de sentimentos

# Projeto FMF - Análise de sentimentos
Referência: <https://artificialcorner.com/p/first-ml-model>

## Contexto

A análise de sentimentos é uma das tarefas mais recorrentes em Processamento de Linguagem Natural (PLN), com aplicações em sistemas de recomendação, monitoramento de reputação de marcas, análise de avaliações de produtos e pesquisas de opinião.

O projeto utiliza esse domínio como contexto para apresentar, de forma didática, as principais etapas de construção de um pipeline de Machine Learning supervisionado para classificação de textos. O contexto escolhido foi o de críticas de filmes do IMDb, em que cada texto possui um sentimento associado.

O trabalho foi inspirado em um notebook introdutório de análise de sentimentos, mas a implementação final amplia a proposta original ao utilizar uma vetorização TF-IDF mais controlada, um baseline simples, modelos ensemble e uma comparação adicional com embeddings semânticos.

## Problema Abordado

O problema tratado é uma tarefa de **classificação binária de texto**. A partir do conteúdo de uma crítica cinematográfica, o modelo deve classificá-la como:

* `positive`: avaliação positiva;
* `negative`: avaliação negativa.

Trata-se de um problema de **aprendizado supervisionado**, pois cada crítica presente no conjunto de dados possui um rótulo conhecido, utilizado durante o treinamento dos modelos.

A pergunta principal do projeto é:

> **Levando em consideração o texto de uma crítica de filme, conseguimos prever automaticamente se ela expressa sentimento positivo ou negativo?**

## Dataset

O projeto utiliza o **IMDB Dataset**, disponibilizado por meio da plataforma Kaggle. O conjunto original contém aproximadamente **50.000 críticas de filmes**, divididas entre avaliações positivas e negativas.

As principais variáveis são:

* `review`: texto completo da crítica cinematográfica;
* `sentiment`: sentimento associado à crítica, classificado como positivo ou negativo.

No pipeline implementado em `pad.ipynb`, o conjunto utilizado depende do arquivo carregado. Quando o arquivo original `IMDB Dataset.csv` é usado, a base contém cerca de **50.000 registros**. Caso seja usado um arquivo limpo em `.parquet`, ou caso alguma linha malformada do `.csv` seja ignorada durante a leitura, esse total pode variar. Por isso, o notebook imprime a quantidade final de registros carregados antes da modelagem.

A distribuição das classes é verificada no próprio notebook. No dataset original do IMDb, as classes positiva e negativa são praticamente equilibradas, o que reduz a necessidade de técnicas artificiais de balanceamento.

Como a base final já estava equilibrada, não foi necessário aplicar undersampling ou oversampling. A estratégia adotada foi preservar a maior quantidade possível de dados e fazer uma divisão estratificada entre treino e teste.

## Tecnologias e Bibliotecas Utilizadas

O projeto utiliza as seguintes bibliotecas:

* `pandas`: leitura, organização e manipulação dos dados;
* `numpy`: apoio a operações numéricas;
* `scikit-learn`: divisão dos dados, TF-IDF, baseline, Stacking, métricas e avaliação;
* `xgboost`: treinamento do modelo XGBoost;
* `lightgbm`: treinamento do modelo LightGBM;
* `matplotlib` e `seaborn`: criação dos gráficos comparativos, curvas ROC, mapa de calor e matriz de confusão;
* `joblib`: salvamento do modelo final e do vetorizador;
* `sentence-transformers`: geração de embeddings semânticos com o modelo `all-MiniLM-L6-v2`.

Para facilitar a execução em outro ambiente, as dependências principais também foram reunidas no arquivo `requirements.txt`.

Como o trabalho será executado no Google Colab, o notebook também procura o dataset em caminhos usados nesse ambiente, como `/content/IMDB Dataset.csv` e `/content/imdb_clean.parquet`. Caso alguma biblioteca não esteja disponível no Colab, ela pode ser instalada com `%pip install -q scikit-learn xgboost lightgbm seaborn joblib sentence-transformers`.

O carregamento do `.csv` também possui uma leitura alternativa para lidar com eventuais linhas malformadas. Se a leitura padrão falhar, o notebook tenta carregar o arquivo ignorando linhas problemáticas, evitando que um erro de formatação interrompa todo o pipeline.

## Estrutura Geral do Pipeline

O projeto segue um pipeline composto pelas seguintes etapas:

1. **Aquisição dos dados:** obtenção do dataset de críticas do IMDb por meio do Kaggle.

2. **Carregamento e verificação:** leitura do arquivo `.csv` ou `.parquet`, conferindo se as colunas `review` e `sentiment` estão presentes.

3. **Limpeza textual:** remoção de tags HTML, caracteres especiais, números, letras maiúsculas e espaços extras.

4. **Codificação dos rótulos:** conversão dos sentimentos para valores numéricos, usando `0` para negativo e `1` para positivo.

5. **Divisão entre treino e teste:** separação dos dados em 80% para treinamento e 20% para teste, com estratificação.

6. **Vetorização textual:** transformação das críticas em representações numéricas utilizando TF-IDF com unigramas e bigramas.

7. **Baseline:** treinamento de uma Regressão Logística para servir como ponto de comparação simples.

8. **Modelagem avançada:** treinamento de XGBoost, LightGBM, Random Forest e Stacking.

9. **Comparação com embeddings:** geração de embeddings semânticos com Sentence-BERT e treinamento de uma Regressão Logística adicional.

10. **Avaliação:** comparação dos modelos por acurácia, precisão, recall, F1-score e AUC-ROC, com escolha do melhor modelo pelo F1-score.

11. **Análise do melhor modelo:** geração do relatório de classificação, matriz de confusão e análise de exemplos classificados incorretamente.

12. **Uso futuro:** salvamento do melhor modelo e criação de uma função para prever o sentimento de novas críticas.

13. **Comunicação:** apresentação de tabela final, gráficos comparativos, mapa de calor das métricas, curvas ROC e gráfico de diferença em relação ao baseline.

## Preparação e Balanceamento dos Dados

A preparação dos dados começa com o carregamento do dataset. O código procura o arquivo em caminhos comuns, como `IMDB Dataset.csv`, `data/IMDB Dataset.csv`, `imdb_clean.parquet` e também caminhos usados no Google Colab. Caso o arquivo não seja encontrado, o notebook informa que o dataset deve ser baixado do Kaggle.

Depois do carregamento, o código verifica a presença das colunas obrigatórias:

* `review`;
* `sentiment`.

Em seguida, registros sem texto ou sem sentimento são removidos. Também é criada a coluna `review_clean`, que contém o texto depois da limpeza.

O balanceamento artificial não foi utilizado no pipeline final. Isso acontece porque a base usada no notebook já apresenta uma distribuição praticamente equilibrada entre as classes. Assim, optou-se por manter a maior parte dos dados disponíveis, em vez de descartar exemplos.

Para preservar esse equilíbrio na avaliação, a separação entre treino e teste foi feita com o parâmetro `stratify=y`.

## Divisão entre Treinamento e Teste

Os dados são separados em:

* 80% para treinamento;
* 20% para teste.

O conjunto de treinamento é utilizado para ajustar o vetorizador TF-IDF e treinar os modelos. O conjunto de teste é reservado para avaliar o desempenho final em críticas que não foram usadas durante o aprendizado.

Como a quantidade final de registros pode variar conforme o arquivo carregado, o notebook imprime o tamanho real dos conjuntos de treino e teste durante a execução. Em todos os casos, a proporção usada é 80% para treino e 20% para teste.

A divisão estratificada é importante porque mantém a proporção entre críticas positivas e negativas nos dois conjuntos.

## Vetorização dos Textos

Como os algoritmos tradicionais de Machine Learning não processam textos diretamente, as críticas precisam ser convertidas em representações numéricas.

Para isso, foi utilizado o `TfidfVectorizer`. O TF-IDF atribui peso maior a termos relevantes em uma crítica, mas que não aparecem de forma excessivamente comum em todo o conjunto de dados.

No notebook, o TF-IDF foi configurado com:

* `max_features=5000`: limita o vocabulário aos 5.000 termos mais importantes;
* `ngram_range=(1, 2)`: considera palavras isoladas e pares de palavras;
* `min_df=5`: remove termos raros demais;
* `max_df=0.8`: remove termos frequentes demais;
* `stop_words='english'`: remove palavras muito comuns em inglês.

O uso de bigramas permite capturar expressões de duas palavras, o que é útil em análise de sentimentos. Expressões como `not good`, `very bad` ou `really enjoyed` podem carregar informação relevante para a classificação.

O vetorizador é ajustado apenas no conjunto de treinamento com `fit_transform()`. No conjunto de teste, usa-se apenas `transform()`, evitando vazamento de informação.

Além do TF-IDF, o notebook acrescenta uma comparação com **embeddings semânticos**. Para isso, é usado o modelo `sentence-transformers/all-MiniLM-L6-v2`, que transforma cada crítica em um vetor numérico capaz de representar melhor o sentido geral do texto.

Essa etapa é mantida em células separadas para facilitar a execução no Google Colab. Os embeddings são salvos em arquivos `.npy`, de modo que não precisam ser recalculados toda vez que o notebook for executado.

## Modelos Avaliados

O projeto compara seis modelos:

* Regressão Logística;
* XGBoost;
* LightGBM;
* Random Forest;
* Stacking;
* Embeddings + Regressão Logística.

A **Regressão Logística** foi usada como baseline. Ela foi escolhida por ser uma abordagem simples, rápida, interpretável e bastante comum em tarefas de classificação de texto com TF-IDF. O papel dela é servir como ponto de comparação para verificar se os modelos mais robustos realmente trazem ganho.

Os outros modelos representam a parte mais original do trabalho em relação à inspiração:

* **XGBoost:** modelo ensemble baseado em gradient boosting;
* **LightGBM:** modelo de gradient boosting eficiente para bases maiores;
* **Random Forest:** conjunto de árvores de decisão treinadas de forma independente;
* **Stacking:** combinação de XGBoost, LightGBM e Random Forest, usando Regressão Logística como meta-modelo.

O modelo **Embeddings + Regressão Logística** tem outro papel: comparar a representação estatística do TF-IDF com uma representação semântica produzida por Sentence-BERT. Assim, o trabalho não apenas testa vários classificadores, mas também compara duas formas diferentes de representar o texto.

## Métricas de Avaliação

O desempenho dos modelos é analisado por meio das seguintes métricas:

* **Acurácia:** proporção total de previsões corretas;
* **Precisão:** proporção de previsões corretas entre os exemplos classificados como positivos;
* **Recall:** proporção de exemplos positivos reais identificados corretamente;
* **F1-score:** média harmônica entre precisão e recall;
* **AUC-ROC:** capacidade do modelo de separar as classes considerando diferentes limiares;
* **Relatório de classificação:** apresenta precision, recall, F1-score e suporte para cada classe;
* **Matriz de confusão:** mostra acertos e erros entre as classes positiva e negativa.

O F1-score foi usado como principal critério de comparação porque resume o equilíbrio entre precisão e recall.

## Otimização de Hiperparâmetros

O pipeline atual não realiza uma busca automática completa de hiperparâmetros com `GridSearchCV`, `RandomizedSearchCV` ou Optuna.

Em vez disso, foram usados hiperparâmetros definidos manualmente para manter o tempo de execução viável e permitir a comparação entre modelos dentro do escopo do trabalho.

Essa decisão é uma limitação do projeto, mas também deixa uma possibilidade clara de melhoria futura: testar combinações de parâmetros de forma sistemática, principalmente para LightGBM, XGBoost e Stacking.

## Características do Projeto

Entre as principais características do projeto estão:

* aplicação prática de classificação de sentimentos;
* uso de uma base grande de críticas do IMDb;
* limpeza textual antes da modelagem;
* divisão treino-teste estratificada;
* vetorização TF-IDF com unigramas e bigramas;
* geração de embeddings semânticos com `all-MiniLM-L6-v2`;
* criação de baseline com Regressão Logística;
* comparação com modelos ensemble;
* comparação entre representação estatística e representação semântica dos textos;
* avaliação por múltiplas métricas;
* tabela final de comparação entre modelos;
* gráficos comparando F1-score e AUC-ROC;
* mapa de calor com o resumo das métricas;
* curvas ROC dos modelos avaliados;
* gráfico de diferença de F1-score em relação ao baseline;
* matriz de confusão para o melhor modelo;
* análise de erros de classificação;
* salvamento do melhor modelo e do vetorizador;
* função para prever o sentimento de novas críticas.

## Limitações Observadas

O projeto apresenta algumas limitações:

* o dataset precisa ser baixado separadamente do Kaggle;
* os hiperparâmetros dos modelos foram definidos manualmente;
* o Stacking tem custo computacional maior do que os modelos individuais;
* o TF-IDF não captura significado semântico profundo, apenas padrões de termos;
* a remoção de stopwords pode afetar expressões com negação, como `not good`;
* os embeddings do `all-MiniLM-L6-v2` podem truncar textos muito longos;
* não foi feito fine-tuning de modelos baseados em Transformers;
* críticas irônicas, ambíguas ou com sentimentos mistos ainda podem ser classificadas incorretamente.

Mesmo com essas limitações, o pipeline apresenta uma evolução clara em relação ao notebook de inspiração, pois inclui baseline, modelos ensemble, embeddings semânticos, tabela comparativa, matriz de confusão e análise de erros.

Link para o colab com a modelagem: <https://colab.research.google.com/drive/13W364xYsE0kYCBHrItTEd0SuB8C3xBYY?usp=sharing>

## Conclusão

O projeto apresenta as principais etapas de um pipeline de Machine Learning aplicado à classificação de sentimentos em textos.

O experimento mostra que é possível prever automaticamente se uma crítica de filme expressa sentimento positivo ou negativo a partir do conteúdo escrito. Para isso, o texto é limpo, transformado em representações numéricas por TF-IDF e também comparado com uma representação semântica por embeddings.

A principal originalidade em relação à inspiração está na comparação entre um baseline simples, modelos ensemble mais robustos e uma alternativa baseada em embeddings. Dessa forma, o trabalho não apenas reproduz a ideia de classificar sentimentos, mas investiga se abordagens diferentes de modelagem e representação textual melhoram a resposta à pergunta inicial.

# Etapas do AGEMC
## ASK — Formulação do Problema
A etapa inicial do projeto parte da pergunta: **“Levando em consideração o texto de uma crítica de filme, conseguimos prever automaticamente se ela expressa sentimento positivo ou negativo?”** Com base nessa questão, o objetivo é desenvolver um modelo capaz de interpretar o conteúdo textual de uma crítica e atribuir a ela uma das duas categorias possíveis de sentimento.

Dessa forma, o projeto é estruturado como um problema de **classificação binária de texto**, no qual a crítica representa a informação de entrada e o sentimento, positivo ou negativo, representa a resposta esperada. Como cada exemplo do conjunto de dados já possui uma classificação associada, o aprendizado ocorre de maneira supervisionada.

## GET — Obtenção dos Dados

Nesta etapa, são reunidos os dados necessários para o desenvolvimento do projeto. A base utilizada é o **IMDB Dataset**, disponibilizado publicamente na plataforma Kaggle e composto por aproximadamente 50.000 avaliações de filmes.

O conjunto apresenta duas variáveis principais: `review`, que armazena o texto escrito pelo usuário, e `sentiment`, que indica se a avaliação foi classificada como positiva ou negativa.

link dos dados: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

No notebook final, o carregamento aceita tanto arquivo `.csv` quanto `.parquet`. Quando o arquivo original `IMDB Dataset.csv` é utilizado, a base contém cerca de 50.000 exemplos. Se for usado um arquivo limpo em `.parquet` ou se alguma linha malformada do `.csv` for ignorada, a quantidade final pode variar. Por isso, o próprio notebook imprime o total carregado.

## EXPLORE — Exploração e Preparação dos Dados

A etapa de exploração tem como objetivo compreender a estrutura do conjunto de dados e prepará-lo para o treinamento dos modelos. Nesse momento, são observadas a distribuição das classes, a quantidade de registros disponíveis, as variáveis presentes e a forma como os textos serão representados numericamente.

### a) Análise da Distribuição das Classes

A base final apresenta distribuição praticamente equilibrada entre críticas positivas e negativas. Por isso, não foi aplicado balanceamento artificial. A decisão foi manter a maior quantidade possível de textos e usar divisão estratificada para preservar a proporção das classes no treino e no teste.

Essa escolha evita o descarte de dados e mantém a avaliação mais próxima do conjunto original.

### b) Separação entre Treinamento e Teste

Os dados são divididos em conjuntos de treinamento e teste por meio da função `train_test_split`, disponibilizada pelo `scikit-learn`.

A divisão reserva 80% dos registros para treinamento e 20% para teste. O parâmetro `stratify=y` é usado para preservar a proporção entre críticas positivas e negativas nos dois conjuntos.

O conjunto de treinamento é utilizado para ajustar o TF-IDF e treinar os modelos. O conjunto de teste é usado apenas no final, para avaliar o comportamento dos modelos diante de críticas não observadas anteriormente.

### c) Conversão dos Textos em Representações Numéricas

Os textos das críticas não podem ser processados diretamente pelos algoritmos clássicos de Machine Learning. Por isso, eles são transformados em vetores numéricos por meio do TF-IDF.

O projeto utiliza `TfidfVectorizer` com limite de 5.000 características, unigramas e bigramas, remoção de termos muito raros e remoção de termos frequentes demais. Com isso, o modelo passa a trabalhar com uma matriz em que cada linha representa uma crítica e cada coluna representa um termo ou par de termos.

Além disso, o projeto inclui uma comparação com embeddings gerados pelo modelo `sentence-transformers/all-MiniLM-L6-v2`. Nessa abordagem, cada crítica também é representada por um vetor semântico, permitindo comparar uma representação baseada em frequência de palavras com uma representação baseada no significado geral do texto.

### d) Prevenção de Vazamento de Dados

O vetorizador TF-IDF aprende o vocabulário e os pesos exclusivamente a partir do conjunto de treinamento. Por esse motivo, utiliza-se `fit_transform()` nos textos de treino e apenas `transform()` nos textos de teste.

Esse procedimento impede que palavras, frequências ou padrões presentes no conjunto de teste influenciem a representação utilizada durante o treinamento.

### e) Matriz de Características

Após a vetorização, a matriz de treinamento possui uma linha para cada crítica do conjunto de treino e até **5.000 características**, de acordo com o limite definido no `TfidfVectorizer`.

Como cada crítica contém apenas uma pequena parcela do vocabulário total, a maior parte dos valores da matriz é igual a zero. Por isso, o `TfidfVectorizer` utiliza uma representação esparsa, que reduz o consumo de memória.

Ao final desta etapa, os dados estão limpos, separados corretamente entre treinamento e teste e representados em formato numérico por TF-IDF. As células de embeddings aparecem separadas no notebook para facilitar a execução e armazenam os vetores gerados em arquivos `.npy`.

## MODEL — Seleção e Treinamento dos Modelos

Nesta etapa, os dados representados numericamente por meio do TF-IDF são utilizados para treinar diferentes algoritmos de classificação supervisionada.

O primeiro modelo treinado é uma **Regressão Logística**, usada como baseline. Esse modelo representa uma solução simples e tradicional para classificação de texto. A partir dele, é possível comparar se modelos mais complexos realmente melhoram o desempenho.

Depois do baseline, o projeto treina os seguintes modelos:

* **XGBoost**;
* **LightGBM**;
* **Random Forest**;
* **Stacking**.

O Stacking combina XGBoost, LightGBM e Random Forest como modelos base, utilizando uma Regressão Logística como meta-modelo. Assim, ele tenta aproveitar diferentes formas de aprendizado e combinar suas previsões.

Depois desses modelos, o notebook acrescenta o modelo **Embeddings + Regressão Logística**. Ele usa os vetores semânticos gerados pelo Sentence-BERT e treina uma Regressão Logística sobre esses embeddings.

Todos os modelos são avaliados no mesmo conjunto de teste e comparados em uma tabela única.

## COMMUNICATE — Avaliação e Comunicação dos Resultados

Nesta etapa, os resultados obtidos pelos modelos são interpretados e apresentados de forma a responder à pergunta definida no início do projeto.

Na execução final do `pad.ipynb`, os resultados obtidos foram:

| Modelo | Acurácia | Precisão | Recall | F1-score | AUC-ROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regressão Logística (baseline) | 0,8906 | 0,8838 | 0,8994 | 0,8916 | 0,9582 |
| Stacking | 0,8677 | 0,8627 | 0,8746 | 0,8686 | 0,9415 |
| LightGBM | 0,8640 | 0,8553 | 0,8762 | 0,8656 | 0,9412 |
| Random Forest | 0,8212 | 0,7847 | 0,8854 | 0,8320 | 0,9048 |
| XGBoost | 0,8149 | 0,7825 | 0,8722 | 0,8249 | 0,9058 |
| Embeddings + Regressão Logística | 0,8234 | 0,8233 | 0,8236 | 0,8234 | 0,9029 |

O melhor desempenho foi obtido pela **Regressão Logística com TF-IDF**, usada como baseline. Esse resultado é importante porque mostra que, para este problema, aumentar a complexidade do modelo não garantiu melhor desempenho. A representação TF-IDF já capturou padrões muito informativos para a separação entre críticas positivas e negativas, e um modelo linear foi suficiente para aproveitar bem esses padrões.

O Stacking e o LightGBM também apresentaram resultados fortes, mas não superaram o baseline. Já a abordagem com embeddings teve desempenho inferior ao TF-IDF com Regressão Logística, o que pode estar relacionado ao fato de o modelo de embeddings usado ser geral e leve, além de críticas longas poderem perder parte do contexto por truncamento.

Além da execução principal, o pipeline também foi testado em uma versão com configurações mais pesadas. Nessa versão, alguns modelos ensemble receberam maior capacidade, como mais estimadores e uma validação interna mais robusta no Stacking. Os resultados foram:

As principais mudanças em relação às configurações iniciais foram:

| Modelo | Configuração inicial | Configuração "pesada" |
| --- | --- | --- |
| XGBoost | `n_estimators=100`, `max_depth=4` | `n_estimators=200`, `max_depth=6` |
| LightGBM | `n_estimators=100` | `n_estimators=200` |
| Random Forest | `n_estimators=100`, `max_depth=10` | `n_estimators=150`, `max_depth=12` |
| Stacking | `cv=2` | `cv=5` |

Essas configurações controlam a complexidade e o custo de treinamento dos modelos:

* **`n_estimators`** define quantas árvores serão treinadas em modelos baseados em árvores, como XGBoost, LightGBM e Random Forest. Ao aumentar esse valor, o modelo passa a ter mais etapas de aprendizado ou mais árvores para combinar. Isso pode melhorar o desempenho, mas também aumenta o tempo de treinamento.
* **`max_depth`** define a profundidade máxima de cada árvore. Árvores mais profundas conseguem capturar relações mais complexas nos dados, mas também podem se ajustar demais ao conjunto de treino, aumentando o risco de overfitting.
* **`cv`** no Stacking define quantas divisões de validação cruzada são usadas para gerar as previsões que treinam o meta-modelo. Um `cv` maior tende a produzir uma estimativa mais estável para o Stacking, mas também torna o treinamento mais demorado, porque os modelos base precisam ser treinados mais vezes.

Assim, a versão pesada foi usada para testar se um aumento controlado de complexidade nos modelos ensemble mudaria a conclusão do pipeline principal.

| Modelo | Acurácia | Precisão | Recall | F1-score | AUC-ROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regressão Logística (baseline) | 0,8906 | 0,8838 | 0,8994 | 0,8916 | 0,9582 |
| LightGBM | 0,8794 | 0,8718 | 0,8896 | 0,8806 | 0,9511 |
| Stacking | 0,8790 | 0,8742 | 0,8854 | 0,8798 | 0,9507 |
| XGBoost | 0,8515 | 0,8328 | 0,8796 | 0,8556 | 0,9338 |
| Random Forest | 0,8259 | 0,7893 | 0,8892 | 0,8363 | 0,9097 |
| Embeddings + Regressão Logística | 0,8234 | 0,8233 | 0,8236 | 0,8234 | 0,9029 |

Essa execução confirmou o comportamento observado no pipeline principal. As configurações mais pesadas melhoraram o desempenho de alguns modelos, principalmente LightGBM, Stacking e XGBoost, mas a **Regressão Logística com TF-IDF continuou sendo o melhor modelo geral**. Portanto, o aumento de complexidade trouxe ganhos pontuais, mas não alterou a conclusão principal do trabalho.

O notebook comunica os resultados por meio de diferentes recursos:

* uma tabela final comparando acurácia, precisão, recall, F1-score e AUC-ROC;
* gráficos comparativos de F1-score e AUC-ROC;
* um mapa de calor resumindo as métricas dos modelos;
* curvas ROC para visualizar a separação entre as classes;
* um gráfico mostrando a diferença de F1-score de cada modelo em relação ao baseline;
* um relatório de classificação do melhor modelo;
* uma matriz de confusão do melhor modelo;
* uma análise de exemplos classificados incorretamente.

A tabela permite comparar o baseline, os modelos ensemble e o modelo baseado em embeddings. Os gráficos ajudam a observar visualmente quais modelos tiveram melhor F1-score e AUC-ROC, e o gráfico de diferença em relação ao baseline mostra se a maior complexidade realmente trouxe ganho. A matriz de confusão mostra quantas críticas positivas e negativas foram classificadas corretamente ou confundidas entre si. A análise de erros ajuda a entender limitações do modelo, principalmente em textos ambíguos, irônicos ou com sentimentos mistos.

De forma geral, o experimento demonstra que a combinação entre limpeza textual, TF-IDF, embeddings e modelos de classificação consegue responder à pergunta inicial. Além disso, a comparação com o baseline e com a representação semântica deixa mais clara a originalidade do trabalho em relação ao notebook de inspiração.

Assim, a resposta ao ASK é positiva: **sim, é possível prever automaticamente se uma crítica de filme expressa sentimento positivo ou negativo a partir do texto**. Porém, o resultado mais relevante da comparação foi perceber que a solução mais simples, Regressão Logística com TF-IDF, teve melhor desempenho que os modelos mais complexos avaliados.

### Objetivo científico do projeto

* Investigar se o conteúdo textual de uma crítica cinematográfica contém informações suficientes para identificar automaticamente o sentimento expresso pelo usuário.
* Avaliar a aplicação de algoritmos de Machine Learning supervisionado em uma tarefa de classificação binária de textos.
* Comparar uma abordagem simples de baseline com modelos ensemble e uma abordagem baseada em embeddings.
* Verificar quais modelos apresentam maior capacidade de generalização para críticas não utilizadas durante o treinamento.
* Analisar se a representação TF-IDF e a representação por embeddings conseguem capturar padrões relevantes para separar avaliações positivas e negativas.

### O que o projeto esperava obter desde o início?

* Construir um modelo capaz de aprender associações entre palavras presentes nas críticas e seus respectivos rótulos de sentimento.
* Identificar qual dos algoritmos avaliados apresenta o melhor desempenho para o problema.
* Obter resultados equilibrados para as classes positiva e negativa.
* Demonstrar, de forma didática, as etapas necessárias para desenvolver um pipeline completo de classificação textual.
* Produzir evidências quantitativas de que o sentimento de uma crítica pode ser estimado a partir de seu conteúdo escrito.

### O que o projeto deseja prever, estimar ou permitir?

* **Prever** se uma nova crítica de filme expressa um sentimento positivo ou negativo.
* **Estimar** a capacidade de diferentes algoritmos de reconhecer padrões linguísticos associados a cada sentimento.
* **Permitir** a classificação automática de grandes volumes de avaliações textuais.
* **Auxiliar** processos de análise de opinião, monitoramento de satisfação e organização de feedbacks.
* **Servir como base** para aplicações futuras mais avançadas, como fine-tuning de Transformers, análise de explicabilidade ou comparação com outros embeddings.

## Linha do Tempo

### Semana 1 — Definição do Tema e Estudo da Referência

Escolha do tema de análise de sentimentos e estudo inicial do projeto de referência. Nessa etapa, o grupo analisou a proposta original e a ideia geral de aplicar Machine Learning supervisionado para classificar críticas de filmes como positivas ou negativas.

### Semana 2 — Entendimento do AGEMC e Preparação Inicial

Organização do trabalho a partir do ciclo AGEMC. Foram definidas a pergunta inicial do projeto, a fonte dos dados, as variáveis principais do dataset e as primeiras etapas de preparação dos textos. Também foi iniciado o entendimento do pipeline de vetorização textual com TF-IDF e divisão entre treino e teste.

### Semana 3 — Definição e Implementação da Originalidade

Definição da contribuição original do projeto, identificando qual etapa do ciclo AGEMC será modificada, ampliada ou adaptada. Nesta semana, a originalidade foi implementada principalmente na etapa de modelagem e comunicação, com a inclusão de baseline, modelos ensemble, embeddings semânticos, tabela comparativa, gráficos, matriz de confusão e análise de erros.

### Semana 4 — Modelagem e Comunicação

Estudo da implementação dos modelos utilizados no projeto de referência, incluindo a análise do processo de treinamento, comparação de desempenho e avaliação das métricas. Também foi aprofundada a etapa de comunicação dos resultados, buscando compreender como as conclusões do projeto são apresentadas e relacionadas ao problema inicialmente definido.

### Semana 5 — Apresentação Final

Preparação e realização de uma apresentação para a turma, reunindo todas as etapas desenvolvidas ao longo do projeto. A apresentação irá abordar o problema investigado, a obtenção e exploração dos dados, a modelagem, a comunicação dos resultados e a contribuição original implementada.
