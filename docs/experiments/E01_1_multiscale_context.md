# E01.1 · Contexto espacial multiescala

O experimento avalia se o contexto de superpixels vizinhos ajuda a separar manchas, superfícies regulares e outros danos. O D01 estimou o limite de IoU do SLIC em 74,82%. O D02 atribuiu 88,08% dos falsos positivos a regiões sem pixels de Stain, e o D03 encontrou ganho limitado com ajuste de limiar. O D04 mostrou dependência dos descritores texturais e diferenças entre as regiões TP e FN. A I01 destacou a dificuldade de cobrir manchas homogêneas.

## Configuração

| Variante | Características |
|---|---|
| B00 | As 15 características e o Random Forest salvos na baseline. |
| E01.1-A | B00 mais `delta_l_r2`, `delta_a_r2`, `delta_b_r2`, `delta_l_r4`, `delta_a_r4`, `delta_b_r4`. |
| E01.1-B | A mais `delta_l_p75_r2`, `delta_l_p75_r4`, `context_l_iqr_r2`, `context_l_iqr_r4`. |

O grafo não direcionado conecta IDs SLIC quando dois pixels vizinhos na horizontal ou na vertical têm IDs distintos. N1 contém os vizinhos imediatos, N2 contém as regiões a até duas arestas e N4 as regiões a até quatro. A própria região fica fora, e cada vizinho entra uma vez. As características originais de contexto usam uma faixa externa produzida pela dilatação de cinco pixels da máscara. O grafo usa **regiões inteiras** e pode cobrir distâncias físicas diferentes entre superpixels.

As médias Lab da vizinhança são ponderadas pela área. Os deltas subtraem a média externa da média da região central. Para as quatro características adicionais, os percentis 25 e 75 são calculados sobre os pixels de luminosidade de toda a vizinhança, com a interpolação linear padrão do NumPy. O IQR é P75 menos P25. Uma vizinhança vazia produz zero em todas as suas características. A extração usa somente RGB e o mapa SLIC.

## Comparabilidade e avaliação

Treino e validação usam os IDs e a ordem de linhas dos NPZ originais. A preparação confere IDs, áreas, contagens de Stain e rótulos antes de anexar as colunas. Na validação, as 15 colunas também são recomputadas para verificar igualdade exata e medir o custo da extração. Os modelos experimentais são treinados nas mesmas linhas binárias, com todos os parâmetros retornados pelo estimador B00, incluindo a semente 42. Todas as regiões, inclusive ambíguas, entram na avaliação por pixel.

Cada variante é avaliada com a decisão padrão, com o limiar fixo do D03 e com um limiar ótimo próprio. A busca usa os cortes distintos, agrupa empates e considera a solução totalmente negativa. Em caso de IoU igual, prevalece o maior limiar. As probabilidades salvas do D02 ancoram o corte exato do B00 após conferência da diferença numérica das novas probabilidades. O ótimo individual é selecionado na validação.

**Definição prévia de mancha homogênea.** A região deve ter ocupação Stain de pelo menos 70% e valores de `glcm_contrast`, `hist_entropy`, `mean_gradient` e `std_l` até os respectivos primeiros quartis dos TP do D04. Os limites são gravados em `configuration.json`. A regra mantém o mesmo conjunto de regiões para todas as variantes. Região predominantemente Erosion significa ocupação de Erosion de pelo menos 70%.

O tempo de inferência por imagem soma carregamento RGB, SLIC, extração original, contexto e classificação. A estimativa de A subtrai os tempos instrumentados dos percentis e do IQR, pois as colunas das duas variantes são extraídas em uma passagem. O tempo de avaliação e gravação dos artefatos é registrado separadamente.

## Execução e artefatos

Na raiz do repositório, com os artefatos da baseline e D02–D04 já disponíveis:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m experiments.evaluate_multiscale_context --phase prepare
.\.venv\Scripts\python.exe -m experiments.evaluate_multiscale_context --phase evaluate
```

As saídas ficam em `results/optimization/E01_1_multiscale_context/`. `datasets/` contém um NPZ de treino e outro de validação, com as 25 colunas. A seleciona o prefixo de 21 colunas, e B usa as 25. `models/` contém apenas os dois estimadores novos. Os CSV registram os cortes individuais, as contagens por imagem e classe de BFDD e as probabilidades e decisões para os seis superpixels de I01.

Os exemplos preservam o conjunto visual fixo do D03, incluindo I01. Incluem também, para cada variante, o maior ganho e a maior perda de IoU na decisão padrão entre as imagens com Stain, com empate por ID. Cada imagem tem uma figura para os padrões e outra para os ótimos individuais.

## Resultados da validação

Os dois modelos novos usaram 417.559 regiões de treino, sendo 409.640 negativas e 7.919 positivas. A preparação produziu 425.202 regiões no treino e 103.621 na validação. O B00 reproduziu as decisões do D02 e as contagens do D03. Os hashes do modelo, manifesto e NPZ originais coincidiram antes e depois da avaliação.

| Variante | Política | Limiar | Precision | Recall | IoU | F1 |
|---|---|---:|---:|---:|---:|---:|
| B00 | Padrão | — | 22,46% | 54,12% | 18,87% | 31,74% |
| B00 | Fixo e ótimo | 0,74048147 | 35,65% | 35,07% | 21,48% | 35,36% |
| E01.1-A | Padrão | — | 24,55% | 54,17% | 20,33% | 33,79% |
| E01.1-A | Fixo | 0,74048147 | 38,00% | 35,38% | 22,43% | 36,64% |
| E01.1-A | Ótimo | 0,69683763 | 34,40% | 39,24% | 22,45% | 36,66% |
| E01.1-B | Padrão | — | 24,44% | 57,00% | 20,64% | 34,21% |
| E01.1-B | Fixo | 0,74048147 | 36,73% | 40,06% | 23,70% | 38,32% |
| E01.1-B | Ótimo | 0,76202650 | 38,60% | 38,43% | 23,85% | 38,52% |

B obteve a melhor IoU global no padrão e no ótimo. A teve a melhor média de IoU por imagem com Stain entre os ótimos (20,48% contra 19,39% de B e 18,42% de B00). Na decisão padrão, A reduziu FP totais em 255.661 e B em 133.489. As duas variantes também reduziram os FP em regiões de ocupação zero, de 2.066.248 para 1.825.114 (A) e 1.919.841 (B). Os FP anotados como Erosion aumentaram de 396.373 para 400.467 e 408.131. As imagens sem mancha com previsão positiva passaram de 51 para 52 e 53. Entre os ótimos, B igualou as 46 imagens da baseline nessa condição e reduziu os FP de ocupação zero para 627.365. Os FP de Erosion no ótimo de B aumentaram de 143.543 para 145.110.

A regra de homogeneidade selecionou 805 regiões. Na decisão padrão, B00 detectou 271, A detectou 281 e B detectou 335. B recuperou 111 regiões homogêneas e perdeu 47 que B00 detectava. Entre todas as regiões predominantemente Stain, B recuperou 200 e perdeu 107 na decisão padrão. Nos ótimos individuais, B detectou 185 homogêneas, frente a 134 em B00.

## I01 e exemplos de ganho e perda

A IoU padrão de I01 passou de 26,12% em B00 para 29,02% em A e 27,42% em B. Nos ótimos individuais, foi de 20,38%, 25,59% e 24,84%. O FP #352 passou de positivo a negativo no padrão das duas variantes, com probabilidade de 0,594 em B00, 0,471 em A e 0,335 em B. Os IDs #127 e #158 continuaram positivos. #353 e #356 continuaram negativos. #764 permaneceu FN mesmo com o aumento de probabilidade de 0,036 para 0,136 em B.

| Variante | Maior ganho de IoU padrão | Maior perda de IoU padrão |
|---|---|---|
| A | `DJI_20250624183233_0077`, 44,81% → 61,46%. Recuperou 5.619 TP líquidos, com 177 FP a mais. | `DJI_20250624185840_0173`, 6,38% → 0%. A máscara perdeu todos os 1.293 TP da baseline, apesar da redução de FP. |
| B | `DJI_20250624190737_0258`, 7,37% → 57,55%. Manteve 122 TP e reduziu FP de 1.521 para 77. | `DJI_20250625185617_0036`, 14,53% → 7,71%. A perda de 3.076 TP superou a redução de 1.504 FP. |

## Custo e limitações

A preparação levou 529,70 s no treino e 213,72 s na validação. O treino levou 22,90 s para A e 30,80 s para B. Os modelos pesam 14.250.440 e 12.686.858 bytes, respectivamente. O tempo médio de inferência por imagem foi de 1,039 s para B00, 1,230 s para A e 1,628 s para B. A fase de treino, avaliação e exportação levou 119,87 s. O experimento gerou 20 figuras de exemplos e dois gráficos.

Os resultados com corte ótimo foram selecionados na própria validação. A regra de homogeneidade caracteriza uma parte dos superpixels predominantemente Stain e não mede a continuidade de manchas inteiras. O ganho global deve ser interpretado junto das regressões em Erosion, nas imagens sem Stain e no custo de inferência.
