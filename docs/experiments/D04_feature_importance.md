# D04 · Importância das características de Stain

O diagnóstico usa o Random Forest salvo e os 132 IDs da validação. A extração reproduz a configuração SLIC e a ordem original das 15 características. Cada região é alinhada ao D02 por imagem, ID original, área e contagem de Stain. As anotações são usadas apenas para agrupamento e avaliação. A matriz resultante, com todas as regiões inclusive as ambíguas, fica em `validation_features.npz`.

## Execução

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m experiments.evaluate_feature_importance
```

Para repetir as análises com a matriz já exportada, use `--reuse-features`. A opção confere a ordem das colunas e o alinhamento com o D02 antes de recalcular as análises.

## Método e critérios de interpretação

- **MDI** usa a redução de impureza registrada no modelo durante o treino. O ranking mostra a soma acumulada e a dispersão entre árvores, e os grupos são somados nas mesmas unidades. A MDI mede o uso interno das características e não mede diretamente sua contribuição para a segmentação de validação.
- **Permutação** usa cinco sementes, de 42 a 46, em cada característica e em cada grupo. A matriz original é preservada. Para grupos, todas as colunas recebem a mesma permutação de linhas. Cada predição avalia as regiões de acordo com suas contagens de pixels, inclusive as ambíguas. As probabilidades perturbadas são avaliadas em dois cortes fixos: a regra de `predict()` com `argmax` na ordem de `classes_`, e `P(Stain) ≥ 0,7404814731187024`. O ranking ordena pela queda média de IoU. Os desvios-padrão são populacionais.
- **Reprodução do D03** usa as probabilidades históricas do D02 para o corte exato. A inferência refeita deve reproduzir as classes originais e concordar nas probabilidades dentro de `1e-12`. Qualquer mudança de lado no corte só é aceita dentro dessa margem e é contabilizada. As referências são verificadas globalmente e por imagem. As permutações usam o limiar exato sem tolerância ou reotimização.
- **Desfechos por região** usam os rótulos do treino: ocupação de Stain até 10% é negativa, pelo menos 70% é positiva e o intervalo intermediário é ambíguo. Os desfechos TP, TN, FP e FN são regionais. Grupos ambíguos são apresentados separadamente, conforme sua previsão.
- **H1 e H2** comparam as medianas e o delta de Cliff nas direções esperadas. H1 compara FN com TP nos seis descritores listados na solicitação. H2 usa os mesmos descritores na direção oposta, comparando FP com TN. O delta de Cliff expressa diferença de distribuição entre os grupos, com valores de −1 a 1. As análises são descritivas e não pressupõem independência entre regiões da mesma imagem.
- **H3** combina as distribuições de `delta_l`, `delta_a` e `delta_b` entre TP e FN com permutações individuais e do grupo de contexto espacial.
- **H4** define como predominante a classe que ocupa **pelo menos 70% da área** do superpixel. A regra é a mesma para Stain, Erosion e Background. Regiões sem uma dessas classes predominantes ficam fora dessa comparação, mas continuam na análise por permutação. Os deltas de Cliff por característica comparam os três pares de classes. Descritores com valores próximos de zero no delta apresentam maior sobreposição de postos, sem implicar equivalência entre classes.
- **Correlação** usa Pearson e Spearman nas 103.621 regiões. O limite descritivo é `|r| ≥ 0,80`. Correlações com uma coluna constante ficam indefinidas. Em I01, os valores estão situados por percentil dentro do próprio desfecho, com posto médio para empates. O intervalo de 5 a 95 é uma referência descritiva de atipicidade.

## Artefatos

Os resultados são gravados em `results/diagnostics/D04_feature_importance/`. As tabelas registram MDI, deltas de permutação individuais e agrupados, repetições, estatísticas por desfecho e classe, matrizes de correlação e os seis superpixels de I01. Os gráficos preservam as unidades das características por painel. Os percentis de I01 usam uma escala comum de 0 a 100.

A importância por permutação mede a dependência do estimador já treinado. Características correlacionadas podem se substituir e reduzir sua importância medida individualmente. A permutação entre regiões pode gerar combinações incomuns. Seus resultados orientam comparações futuras e devem ser confirmados com ablação controlada.

## Resultados da execução

As referências reproduziram todas as contagens, globalmente e por imagem. A IoU foi de 18,87% na decisão padrão e de 21,48% no corte D03. A maior diferença numérica entre as probabilidades refeitas e as do D02 foi `6,66e-16`, sem mudança de lado no limiar. A matriz tem 103.621 linhas e 15 colunas. A extração e a gravação do NPZ levaram 321,02 s, as 100 permutações levaram 20,34 s e a execução inteira levou 356,72 s.

Na MDI, `glcm_asm` ficou em primeiro (15,54%), seguida por `hist_entropy` (9,98%) e `mean_gradient` (9,10%). A MDI por grupo foi de 36,50% para GLCM, 32,10% para cor CIELab, 12,32% para contexto, 9,98% para histograma e 9,10% para gradiente. A permutação confirmou `glcm_asm` como principal característica nos dois cortes. Sua queda média de IoU foi de 11,78 p.p. no padrão e 17,51 p.p. no D03. `glcm_homogeneity` ficou em segundo nos dois cortes.

| Grupo | Queda de IoU padrão (p.p.) | Queda de IoU D03 (p.p.) |
|---|---:|---:|
| Textura GLCM | 14,26 | 19,63 |
| Cor CIELab | 6,17 | 14,78 |
| Histograma | 2,30 | 14,91 |
| Contexto espacial | 1,73 | 3,58 |
| Gradiente | 1,14 | 13,13 |

### Hipóteses e I01

- **H1:** entre 915 FN e 1.505 TP, os seis indicadores foram na direção esperada. FN tiveram menor contraste, entropia, gradiente e `std_l`, e maior homogeneidade e ASM. Os deltas de Cliff de FN menos TP variaram de −0,564 a −0,491 nos indicadores de intensidade e irregularidade e foram +0,610 para homogeneidade e +0,643 para ASM.
- **H2:** entre 5.402 FP e 93.623 TN, os seis indicadores também foram na direção esperada. FP tiveram maior irregularidade textural pelos indicadores escolhidos. Entropia e gradiente tiveram deltas de Cliff de +0,695 e +0,647, respectivamente, para FP menos TN. As 2.176 regiões ambíguas ficaram em grupos próprios, com 699 previsões positivas e 1.477 negativas.
- **H3:** `delta_l` liderou o contexto em ambos os cortes, com queda de 1,01 e 2,34 p.p. na IoU por permutação individual. A mediana de TP menos FN foi −1,579 em `delta_l`, +0,054 em `delta_a` e +0,167 em `delta_b`. O contexto conjunto teve queda de 1,73 e 3,58 p.p.
- **H4:** foram comparadas 2.420 regiões Stain, 2.378 Erosion e 94.301 Background, todas com pelo menos 70% de ocupação da classe respectiva. O modelo previu Stain em 834 das regiões Erosion (35,07%) e em 4.020 das regiões Background (4,26%). Erosion e Stain tiveram sobreposição nos postos de homogeneidade e correlação GLCM (`|δ| < 0,10`). Em comparação, as diferenças em `mean_a` (δ = +0,585) e `mean_b` (δ = +0,640) foram maiores. A similaridade observada depende do descritor.
- **I01:** #127 e #158 não ficaram fora do intervalo de percentis 5–95 de TP em nenhuma feature. O FP #352 ficou nos percentis 97,6 de `std_a` e 96,6 de `glcm_correlation` entre FP. O FN #764 ficou no percentil 1,6 de `mean_l` entre FN. #353 e #356 tiveram valores altos de `delta_b` dentro de TN.

### Correlação e próximos experimentos

O critério de 0,80 identificou oito pares fortes em Pearson e 14 em Spearman. Nos pares de interesse, Pearson/Spearman foram −0,613/−0,926 para contraste e homogeneidade GLCM, −0,848/−0,959 para ASM e entropia, +0,847/+0,890 para gradiente e contraste, e +0,837/+0,960 para `std_l` e entropia.

Para futuras ablações, compare GLCM como bloco e combinações de ASM, homogeneidade e entropia. `std_a` e `delta_a` são candidatos a testes de remoção isolada por sua importância relativamente baixa nos dois cortes. `std_l` e `glcm_contrast` merecem avaliação nos dois cortes, pois seu embaralhamento melhorou a IoU no padrão e a reduziu no D03. Para inclusão de novas características, teste contexto de luminosidade em mais de uma escala e verifique separadamente a confusão com Erosion, acompanhando precisão, recall e IoU. Nenhuma característica foi removida neste diagnóstico.
