# ANFIS

O ANFIS (Adaptive-Network-based Fuzzy Inference System) foi proposto por J.-S. R. Jang em 1993. Ele combina redes neurais com sistemas de inferência fuzzy do tipo Takagi-Sugeno, de modo que os parâmetros do sistema fuzzy podem ser aprendidos a partir de dados.

## Arquitetura

A arquitetura do ANFIS tem cinco camadas. A camada 1 faz a fuzzificação das entradas com funções de pertinência. A camada 2 calcula a força de disparo de cada regra aplicando um operador de conjunção (uma t-norma), geralmente o produto. A camada 3 normaliza as forças de disparo. A camada 4 calcula a saída de cada regra, uma função linear das entradas ponderada pela força normalizada. A camada 5 soma todas as saídas e produz a saída final do modelo.

## Treinamento

O ANFIS usa um algoritmo de aprendizado híbrido. Na passada direta, os parâmetros dos consequentes são ajustados pelo método dos mínimos quadrados. Na passada reversa, os parâmetros das funções de pertinência são ajustados por gradiente descendente com retropropagação do erro.

## Limitações

Com uma partição em grade, o número de regras cresce exponencialmente com o número de entradas: com m funções de pertinência por entrada e n entradas, o modelo tem m elevado a n regras. Isso limita o uso do ANFIS em problemas com muitas variáveis e aumenta o consumo de memória.
