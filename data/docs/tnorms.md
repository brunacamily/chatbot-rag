# T-normas

Uma t-norma é uma função T de [0,1] x [0,1] em [0,1] que é comutativa, associativa, monótona e tem o número 1 como elemento neutro. Em sistemas fuzzy, a t-norma generaliza o conectivo "E" da lógica clássica e é usada para combinar os antecedentes das regras.

## Exemplos clássicos

- Mínimo (Gödel): T(a, b) = min(a, b).
- Produto algébrico: T(a, b) = a * b.
- Łukasiewicz: T(a, b) = max(0, a + b - 1).
- Drástica: T(a, b) = b se a = 1, T(a, b) = a se b = 1, e 0 nos demais casos.

Essas t-normas obedecem à ordem: drástica <= Łukasiewicz <= produto <= mínimo. O mínimo é a maior t-norma e a drástica é a menor.

## T-normas paramétricas

Existem famílias de t-normas com um ou mais parâmetros que controlam a forma do operador, como Hamacher, Yager, Frank, Dombi e Schweizer-Sklar. Ajustar esse parâmetro permite que o modelo escolha, a partir dos dados, o comportamento da conjunção mais adequado ao problema.
