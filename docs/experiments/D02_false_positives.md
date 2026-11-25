# D02 · Origem dos falsos positivos de Stain

O D02 descreve os falsos positivos da baseline nas 132 imagens de validação. Ele usa o modelo salvo, os parâmetros SLIC e as 15 características registradas. A inferência ocorre antes da leitura das anotações. A classe prevista vem de `predict()` e as probabilidades vêm de `predict_proba()`. Os intervalos de probabilidade caracterizam as decisões existentes. O experimento não busca um novo limiar.

## Execução

Na raiz do projeto, com os artefatos da baseline e do D01 disponíveis:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m experiments.evaluate_false_positives
```

Os artefatos são gravados em `results/diagnostics/D02_false_positives/`. A execução confere cada imagem contra as contagens da baseline. O cruzamento com D01 usa `(image_id, region_id)` e exige igualdade de área, contagens e ocupação. O CSV `region_statistics.csv` contém todas as regiões, suas contagens, probabilidades e decisões originais para uso posterior no D03.

## Convenções de análise

- A taxa de FP por classe original usa a quantidade de pixels anotados naquela classe como denominador. A taxa por imagem usa `FP / (FP + TN)`. A participação nos FP usa o total de FP do agrupamento ou escopo indicado.
- A ocupação usa contagens inteiras nos limites. Zero e 100% têm categorias próprias, 60% entra em `(50%, 60%]` e 70% entra em `[70%, 80%]`. A média e a mediana das probabilidades nos grupos de ocupação e rótulo de treino se referem às regiões previstas como Stain do grupo.
- Os intervalos de probabilidade incluem o extremo inferior e excluem o superior, exceto o último, que inclui 1,00. Alta probabilidade significa `P(Stain) ≥ 0,90`. O ranking usa os FP dessa faixa, inclusive em regiões parcialmente ocupadas. O relatório global separa as regiões de alta probabilidade com ocupação exatamente zero.
- Razões com denominador zero retornam `0.0`. Média ou mediana de probabilidades sem regiões selecionadas é registrada como `null`.
- Os FP em regiões também selecionadas pelo Oracle Max-IoU são erros aceitos por aquela solução para maximizar a cobertura e a IoU global. Sua presença nessa seleção não demonstra que sejam inevitáveis.

## Exemplos e limites de interpretação

Os exemplos incluem I01 (`DJI_20250624181834_0007`) e até duas imagens dos rankings de FP absolutos, FP em imagens sem Stain e FP de alta probabilidade. Empates usam o ID como critério secundário. Imagens selecionadas mais de uma vez geram um único PNG e mantêm todos os critérios no relatório global.

A anotação `BACKGROUND` reúne pixels sem uma das classes de dano escolhidas pelo BFDD. Ela não identifica semanticamente paredes, janelas, reflexos ou equipamentos. As tabelas quantificam onde os FP ocorreram em relação às classes e à ocupação. Qualquer atribuição a elementos da fachada depende de evidência visual específica.

## Resultados da execução

A execução reproduziu as contagens da baseline em todas as 132 imagens. São 679.462 TP, 39.652.337 TN, 2.345.849 FP e 576.112 FN, em 43.253.760 pixels. O cruzamento confirmou as estatísticas dos 103.621 superpixels do D01. A análise levou 118,77 s, dos quais 109,82 s foram leitura RGB e inferência com probabilidades. Foram gravados seis gráficos e seis exemplos.

| Classe original | Pixels | FP | Participação nos FP | Taxa de FP na classe |
|---|---:|---:|---:|---:|
| Background | 39.300.186 | 1.699.543 | 72,45% | 4,32% |
| Crack | 737.292 | 126.084 | 5,37% | 17,10% |
| Peeling | 289.584 | 54.682 | 2,33% | 18,88% |
| Hollow Area | 431.594 | 69.167 | 2,95% | 16,03% |
| Erosion | 1.239.530 | 396.373 | 16,90% | 31,98% |

Regiões com ocupação zero responderam por 2.066.248 FP (88,08%). Dentro desse grupo, 311 regiões foram previstas como Stain com probabilidade de pelo menos 0,90, gerando 114.013 FP. Pelo critério de treino, regiões negativas geraram 91,32% dos FP, ambíguas geraram 7,03% e positivas geraram 1,65%.

As 75 imagens com Stain contribuíram com 1.523.282 FP (64,94%). Das 57 imagens sem Stain, 51 tiveram máscara positiva. Essas 57 imagens contribuíram com 822.567 FP (35,06%). Os cinco, dez e vinte primeiros IDs do ranking absoluto concentraram, respectivamente, 17,43%, 28,25% e 45,13% dos FP.

A seleção do Oracle Max-IoU também inclui regiões que contêm 93.878 FP da baseline (4,00%). Os 2.251.971 FP restantes (96,00%) pertencem a regiões não selecionadas pelo oráculo. Esses dados mostram que a maior parte dos erros ocorreu fora da seleção de regiões que maximiza a IoU, mas a distribuição, por si só, não identifica quais características visuais levaram o modelo a errar.
