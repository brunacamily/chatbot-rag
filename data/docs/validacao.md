# Validação e comparação de modelos

## Validação cruzada k-fold

Na validação cruzada k-fold, os dados são divididos em k partes (folds). Cada parte é usada uma vez como conjunto de teste, enquanto as outras k-1 partes treinam o modelo. A métrica final é a média dos k resultados. Valores comuns são k = 5 e k = 10. Isso dá uma estimativa de desempenho mais estável do que uma única divisão treino/teste. A versão estratificada preserva a proporção das classes em cada fold.

## Busca em grade de hiperparâmetros

A busca em grade (grid search) testa todas as combinações de uma lista de valores para cada hiperparâmetro e escolhe a combinação com melhor desempenho. Para evitar vazamento de informação, a escolha dos hiperparâmetros deve usar dados de validação separados dos dados usados na avaliação final.

## Comparação estatística entre classificadores

Para comparar vários classificadores em vários datasets, Demšar (2006) recomenda o teste de Friedman, um teste não paramétrico baseado nos postos (ranks) de cada algoritmo em cada dataset. A hipótese nula é que todos os algoritmos têm desempenho equivalente. Quando ela é rejeitada, aplica-se um teste post-hoc, como o de Nemenyi, que usa uma diferença crítica entre os postos médios. Para comparar apenas dois classificadores pareados em vários datasets, usa-se o teste de Wilcoxon dos postos sinalizados.
