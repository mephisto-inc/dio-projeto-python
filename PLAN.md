# Entendendo o Desafio

Agora é a sua hora de praticar, aprender fazendo e treinar modelos que detectam fraude em transações reais de cartão de crédito, um problema em que a fraude é rara e a acurácia engana.

**Base de Dados:**  
[https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)

Nas Aulas, a Expert Isadora percorre o pipeline inteiro num notebook em **Python**, da coleta dos dados à explicação do modelo, comparando algoritmos pelas métricas que importam em fraude. Na Aula 1, ela passa esse link direto para o `pd.read_csv`, sem baixar o arquivo.

## O Que Criar

Um notebook que carrega o dataset, trata o desbalanceamento, treina e compara modelos e explica o que levou uma transação a ser marcada como fraude. Na base, `Time` marca o momento, `Amount` o valor e `Class` a resposta, com 1 para fraude e 0 para normal. As variáveis `V1` a `V28` foram transformadas por PCA para proteger a privacidade.

O ponto central é o desbalanceamento: 99,8% das transações são normais e só 0,17% são fraude. Um modelo que responde "não é fraude" para tudo acerta quase sempre e não serve para nada, por isso o foco sai da acurácia e vai para a classe de fraude.

Nessa classe, o recall mede quantas fraudes reais o modelo encontrou, a precisão mede quantas das marcadas como fraude eram mesmo fraude, e o F1 equilibra as duas.

## Como Fazer

Não existe repositório base. Você monta o seu notebook do zero, seguindo o caminho da Expert, que trabalha no **Google Colab**:

1. **Explore.** Carregue a base com o **pandas**, veja as primeiras linhas e meça a proporção de fraudes com `value_counts(normalize=True)`;
2. **Prepare.** Crie o log do valor com `np.log1p`, padronize com o `StandardScaler` do **scikit-learn** e separe treino e teste com `stratify`, para manter a proporção de fraudes nos dois;
3. **Treine.** Comece pela regressão logística como baseline e compare com o Random Forest, dando mais peso à fraude com `class_weight='balanced'`, e com o **XGBoost**;
4. **Avalie.** Leia o relatório de classificação, a curva ROC e a curva de precisão e recall, a mais útil num problema desbalanceado;
5. **Ajuste.** Junte padronização e modelo num `Pipeline`, baixe o limiar de decisão de 0,5 para 0,3 para pegar mais fraudes e teste hiperparâmetros com o `GridSearchCV`, escolhendo pelo recall;
6. **Explique.** Veja a importância das variáveis e use o **SHAP** para mostrar o que pesou em cada decisão.

## Ideias para Evoluir

- Treinar com undersampling e com oversampling e comparar o recall, porque a regra da Expert para escolher entre os dois é testar;
- Levar o Random Forest para o Pipeline com ajuste de limiar, como a Expert sugere;
- Ampliar a busca de hiperparâmetros do `GridSearchCV`;
- Criar variáveis de comportamento ao longo do tempo, como transações em sequência;
- Testar outros modelos de classificação e comparar todos pelo recall.

Se você ainda está começando, tudo bem. Um modelo simples, com as métricas certas e bem explicadas, vale mais que cinco algoritmos comparados só pela acurácia.

## Uma Ajuda durante o Caminho

Quando as métricas parecerem boas demais, o [DIO Agent](https://github.com/digitalinnovationone/dio-agent) ajuda a interpretar:

```
Estou treinando modelos para detectar fraude em cartão de crédito, numa base em que só 0,17% das transações são fraude.
Meu modelo passou de 99% de acurácia, mas não sei se ele é bom de verdade.
Me ajude a entender por que a acurácia engana aqui e como ler recall, precisão e F1 da classe de fraude.

Importante: não quero uma resposta pronta para copiar.
Quero entender o processo e construir minha própria solução.
```

## O Que Entregar

Suba o notebook para um repositório público na sua conta do GitHub. No `README.md`, explique:

- O problema, e por que o desbalanceamento muda a forma de avaliar;
- O que você fez na preparação dos dados;
- A comparação entre os modelos, com recall, precisão e F1 da classe de fraude;
- O limiar de decisão escolhido e o que o SHAP mostrou;
- O que você mudou em relação ao que a Expert fez.

Deixe as saídas salvas no notebook, com as tabelas e as curvas. Evidência de que o modelo roda é o que mais pesa em um portfólio.

Antes de submeter, confira:

- O notebook roda do início ao fim, sem célula quebrada;
- O dataset é lido pelo link da Base de Dados e não vai para o repositório, porque o arquivo passa do limite de 100 MB do GitHub;
- Todo arquivo citado no README existe e tem conteúdo;
- O link enviado é o do repositório, não o de um arquivo dentro dele;
- O repositório está na sua conta e público;
- O nome do repositório ficou legível, em minúsculas e sem acento.

## Resultado Esperado

Ao final, você terá um projeto de detecção de fraude com dados reais, do tratamento à explicação do modelo. O repositório mostra o que mais conta nessa área: escolher a métrica certa para um problema desbalanceado e justificar cada decisão do modelo.

Bons estudos e bom projeto 🚀
