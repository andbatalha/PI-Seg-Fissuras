# D03 · Limiar de decisão do Random Forest

O experimento usa as decisões originais e as probabilidades das 103.621 regiões exportadas pelo D02, nas 132 imagens de validação. As métricas vêm da soma das contagens de pixels das regiões. O SLIC é reexecutado apenas para reconstruir os exemplos visuais. A extração de características e o classificador ficam fora da execução.

## Execução

Na raiz do projeto, com os artefatos do D02 e da baseline disponíveis:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m experiments.evaluate_thresholds
```

## Seleção e convenções

- A baseline usa a decisão de `predict()` armazenada no CSV, inclusive em eventuais empates. A curva usa `P(Stain) ≥ limiar`, inclusive no ponto 0,50. Toda região é positiva em 0,00. Em 1,00, só as regiões com probabilidade exatamente 1 são positivas.
- A grade inclui 101 pontos de 0,00 a 1,00. A busca exata também inclui todos os valores distintos de probabilidade e o caso totalmente negativo. Probabilidades iguais entram ou saem juntas. A linha `all_negative` do CSV tem limiar vazio e representa esse caso especial.
- Nos empates, escolhe-se o maior limiar, incluindo o caso totalmente negativo acima de todos os cortes. As restrições de Precision e Recall são verificadas por contagens de pixels. Para cada restrição, maximiza-se a IoU. Uma restrição inviável é registrada como `null`.
- Com as mesmas contagens globais, `F1 = 2 × IoU / (1 + IoU)`. A seleção de IoU e F1 usa comparações de razões inteiras e o mesmo desempate.
- Razões com denominador nulo retornam zero. A comparação de variação de IoU por imagem considera apenas imagens com Stain. Para imagens sem Stain, registra-se quantas máscaras têm pixels positivos. O CSV de regiões possui contagens de Stain e não Stain, sem a composição multiclasse por região necessária para reclassificar os FP por classe original.

## Artefatos e amostras

Os artefatos são gravados em `results/diagnostics/D03_threshold_analysis/`. `threshold_metrics.csv` contém a grade, a busca exata e a baseline, com colunas globais e separadas por presença de Stain. A lista de melhores cortes fica em `optimal_thresholds.json`. As comparações por imagem e dos seis superpixels de I01 são gravadas em CSV separados.

Os exemplos incluem I01, até duas imagens com os maiores ganhos de IoU, até duas com as maiores perdas e até duas imagens sem Stain com mais FP eliminados. Empates na seleção visual são desfeitos pelo ID. Antes de gravar cada PNG, o experimento verifica se os IDs, áreas, ocupação e contagens reconstruídas correspondem aos dados tabulares.

O corte ótimo foi selecionado nesta validação. Uma estimativa imparcial de desempenho em imagens desconhecidas exige avaliação independente.

## Resultados

Foram analisadas 52.112 probabilidades distintas, além da solução totalmente negativa. As decisões originais reproduziram as contagens salvas em todas as imagens. Não houve probabilidade igual a 0,50 neste CSV. Assim, `P ≥ 0,50` e a baseline produziram decisões idênticas nos dados analisados.

| Decisão | Precision | Recall | IoU | F1 |
|---|---:|---:|---:|---:|
| Baseline `predict()` | 22,46% | 54,12% | 18,87% | 31,74% |
| 0,60 | 26,36% | 46,90% | 20,30% | 33,75% |
| 0,70 | 32,11% | 38,39% | 21,19% | 34,97% |
| 0,80 | 41,33% | 28,20% | 20,14% | 33,53% |
| 0,90 | 54,63% | 13,96% | 12,51% | 22,23% |
| Ótimo 0,7404814731187024 | 35,65% | 35,07% | 21,48% | 35,36% |

O mesmo corte maximizou IoU e F1. Ele eliminou 1.550.783 FP, acrescentou 239.078 FN e reduziu a área prevista em 1.789.861 pixels. Os FP em regiões de ocupação zero caíram de 2.066.248 para 666.421. Entre as 75 imagens com Stain, 23 melhoraram em IoU, 45 pioraram e sete ficaram iguais. Entre as 57 imagens sem Stain, as máscaras positivas passaram de 51 para 46.

Em I01, a IoU caiu de 26,12% para 20,38%. O corte corrigiu o FP do superpixel #352, manteve #127 e #158 como Stain e deixou #353, #356 e #764 negativos. Na imagem completa, Precision subiu de 45,23% para 59,82% e Recall caiu de 38,20% para 23,62%.

A execução levou 12,39 s incluindo a exportação de sete gráficos e sete exemplos. Os 26 testes passaram, incluindo comparação da busca exata com avaliação exaustiva em 70 conjuntos sintéticos.
