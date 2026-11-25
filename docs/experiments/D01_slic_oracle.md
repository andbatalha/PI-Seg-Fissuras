# D01 · Limite teórico da segmentação SLIC

O experimento mede a melhor segmentação de Stain possível quando cada superpixel recebe uma única classe. As máscaras dos autores orientam a seleção de regiões. Os resultados dos oráculos são referências diagnósticas da partição SLIC, sem medir a capacidade de inferir manchas em imagens novas.

## Configuração e execução

O comando lê os 132 IDs de validação de `results/region_full/split.json` e a configuração SLIC de `random_forest.joblib`. O modelo é carregado apenas para recuperar seus metadados. O relatório de validação por pixel fornece as métricas da baseline. Na raiz do projeto, execute:

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_slic_oracle
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Critérios

- **Majority:** uma região é positiva quando contém mais pixels Stain que pixels não Stain. Em caso de empate, fica negativa. A escolha minimiza o total de erros por pixel, porque cada região contribui independentemente para essa contagem.
- **Max-IoU:** reúne as regiões de toda a validação e ordena pela ocupação real decrescente. Para cada prefixo, calcula `TP / (Stain de toda a validação + FP)`. Entre prefixos com a mesma IoU, mantém o menor. O limiar registrado corresponde à menor ocupação entre as regiões selecionadas.

Para justificar a busca por prefixos, considere uma solução com IoU `J`. Adicionar uma região fora dela aumenta a IoU quando os pixels Stain da região excedem `J` vezes seus pixels não Stain. Remover uma região selecionada melhora a IoU sob a desigualdade inversa. Logo, em um ótimo, as regiões acima de `J / (1 + J)` em ocupação entram e as regiões abaixo ficam de fora. As regiões na igualdade são indiferentes. Isso garante que existe um prefixo ótimo na ordenação por ocupação, mesmo com tamanhos diferentes de superpixel.

Se uma imagem não contém Stain, os dois critérios produzem máscara negativa. Precision, Recall, IoU e F1 seguem a convenção do projeto de retornar zero quando seus denominadores são nulos. Se toda a validação estiver sem Stain, o Max-IoU seleciona zero regiões e registra o limiar como `null`.

## Artefatos

`results/diagnostics/D01_slic_oracle/` recebe a configuração, as métricas globais e individuais, as estatísticas por região em CSV, o histograma de ocupação e os exemplos PNG. O histograma usa intervalos fechados à esquerda e abertos à direita, exceto o último, que inclui 100%. As figuras incluem exemplos positivos com IoU baixa, mediana e alta no Max-IoU global, uma imagem sem Stain e, quando houver correspondência única das quatro contagens da baseline, a imagem de I01.

O tempo de processamento de cada variante inclui preparação compartilhada, seleção e reconstrução com avaliação. No relatório por imagem, o tempo inclui preparação e reconstrução com avaliação; a seleção global é cronometrada uma vez no relatório global. O tempo de parede também inclui a gravação das figuras e dos arquivos tabulares. Os tempos dos oráculos e da baseline correspondem a operações diferentes.

## Resultado da execução

Foram avaliados 43.253.760 pixels em 132 imagens, das quais 57 não continham Stain. Os parâmetros registrados foram `n_segments=800`, `compactness=10`, `sigma=1` e `start_label=1`.

| Método | Accuracy | Precision | Recall | IoU | F1 |
|---|---:|---:|---:|---:|---:|
| Random Forest · baseline salva | 93,24% | 22,46% | 54,12% | 18,87% | 31,74% |
| Oracle Majority | 99,17% | 86,97% | 83,83% | 74,48% | 85,37% |
| Oracle Max-IoU | 99,15% | 84,37% | 86,86% | **74,82%** | 85,60% |

Dos 103.621 superpixels, 97.503 não contêm Stain, 987 têm 100% de ocupação e 5.131 têm ocupação parcial. O Max-IoU selecionou 3.164 regiões, com ocupação mínima de 42,8082%. A diferença até a baseline é de 55,95 pontos percentuais de IoU. O limite mede o efeito da representação por regiões inteiras nesta partição e nestas anotações. A diferença até a baseline indica espaço para melhorar a seleção de regiões, sem estimar quanto um classificador pode recuperar em imagens novas.

O preparo compartilhado levou 111,69 s. O processamento total por variante foi de 113,45 s para Majority e 113,51 s para Max-IoU, incluindo o mesmo preparo em cada tempo. A execução completa, com a exportação, levou 129,50 s.

Foram gerados cinco exemplos. `DJI_20250624181834_0007` corresponde unicamente às quatro contagens individuais transcritas em I01. O vínculo é uma identificação por correspondência no relatório salvo, pois I01 não registra o ID diretamente.
