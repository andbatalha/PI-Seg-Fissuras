# Análise e preparação do dataset BFDD

Este documento registra as verificações e experimentos realizados sobre o dataset BFDD antes do desenvolvimento dos algoritmos de processamento e segmentação.

## 1. Estrutura do dataset

O dataset utilizado é o BFDD. Foram identificadas cinco pastas principais:

- `IR/`: 838 imagens
- `RGB/`: 838 imagens
- `Label/`: 838 máscaras numéricas
- `Label_color/`: 838 máscaras coloridas
- `Label_backup_7classes_20260125/`: 838 máscaras

Total: 4190 arquivos de imagem, correspondentes a 838 amostras.

As imagens RGB possuem dimensão:

`512 × 640 × 3`

As labels numéricas possuem dimensão:

`512 × 640`

As labels são armazenadas como imagens `uint8`.

---

## 2. Classes identificadas

A inspeção das máscaras numéricas mostrou a utilização dos valores de 0 a 5.

O mapeamento adotado no projeto foi:

| ID | Classe |
|---:|---|
| 0 | BACKGROUND |
| 1 | CRACK |
| 2 | PEELING |
| 3 | HOLLOW_AREA |
| 4 | EROSION |
| 5 | STAIN |

Durante a inspeção visual do dataset, houve maior segurança na identificação das classes:

- `BACKGROUND`
- `CRACK`
- `HOLLOW_AREA`

As distinções visuais entre `PEELING`, `EROSION` e `STAIN` mostraram-se menos evidentes em algumas imagens, havendo casos visualmente ambíguos.

---

## 3. Teste de carregamento

Foram implementadas e testadas funções separadas para carregar imagens RGB e labels.

Resultados observados:

- Imagem RGB: `(512, 640, 3)`
- Label: `(512, 640)`
- Tipo das labels: `uint8`

Em uma das labels utilizadas para teste foram encontrados os valores:

`[0, 1, 2, 5]`

Isso confirmou que uma única imagem pode possuir múltiplas manifestações patológicas simultaneamente.

---

## 4. Geração de máscaras binárias

Foi implementada uma função para transformar a label multiclasses em uma máscara binária para uma classe específica.

Exemplo para `CRACK`:

- pixel pertencente à classe CRACK → 1
- qualquer outro pixel → 0

Em uma amostra testada, a máscara de CRACK apresentou 620 pixels positivos.

Ao gerar uma máscara para EROSION na mesma imagem, o resultado foi completamente zero, pois essa classe não estava presente naquela amostra.

Esse teste confirmou que as máscaras específicas de cada classe podem ser extraídas corretamente da label original.

---

## 5. Indexação das classes presentes

Foi construído um índice relacionando cada imagem às classes presentes em sua label.

Estrutura conceitual:

`nome_da_imagem -> [classes presentes]`

O índice contém as 838 imagens do dataset.

A classe `BACKGROUND` aparece em todas as imagens e não é considerada uma manifestação patológica.

---

## 6. Frequência das classes por imagem

Foi contado o número de imagens que possuem pelo menos um pixel de cada classe.

| Classe | Imagens | Percentual aproximado |
|---|---:|---:|
| BACKGROUND | 838 | 100% |
| CRACK | 833 | 99,4% |
| PEELING | 188 | 22,4% |
| HOLLOW_AREA | 383 | 45,7% |
| EROSION | 584 | 69,7% |
| STAIN | 472 | 56,3% |

Importante: esses valores representam a quantidade de **imagens contendo a classe**, e não a quantidade de pixels pertencentes a cada classe.

Foi observado que CRACK está presente em 833 das 838 imagens, portanto somente 5 imagens não apresentam fissuras.

---

## 7. Combinações de classes

Como uma imagem pode possuir várias manifestações simultaneamente, as imagens foram agrupadas pela combinação exata de classes presentes, desconsiderando BACKGROUND.

Foram encontradas 19 combinações:

| Combinação | Imagens |
|---|---:|
| CRACK + PEELING + STAIN | 19 |
| CRACK + STAIN | 92 |
| CRACK + EROSION + STAIN | 133 |
| SEM DANOS | 2 |
| CRACK | 68 |
| CRACK + HOLLOW_AREA + EROSION | 109 |
| CRACK + HOLLOW_AREA + EROSION + STAIN | 128 |
| CRACK + PEELING + HOLLOW_AREA + EROSION + STAIN | 56 |
| CRACK + PEELING | 20 |
| PEELING | 2 |
| CRACK + HOLLOW_AREA | 26 |
| CRACK + HOLLOW_AREA + STAIN | 12 |
| CRACK + EROSION | 79 |
| CRACK + PEELING + EROSION + STAIN | 27 |
| CRACK + PEELING + EROSION | 12 |
| CRACK + PEELING + HOLLOW_AREA + EROSION | 39 |
| CRACK + PEELING + HOLLOW_AREA | 8 |
| CRACK + PEELING + HOLLOW_AREA + STAIN | 5 |
| EROSION | 1 |

A soma dos grupos resulta nas 838 imagens.

As cinco imagens sem CRACK correspondem exatamente a:

- 2 imagens sem danos;
- 2 imagens contendo apenas PEELING;
- 1 imagem contendo apenas EROSION.

Isso serviu também como teste de consistência da indexação.

---

## 8. Primeiro experimento de divisão do dataset

Inicialmente foi considerada uma divisão aleatória de aproximadamente:

- 80% para desenvolvimento;
- 20% para teste.

A divisão foi realizada dentro de cada combinação de classes, de forma a preservar aproximadamente a distribuição multilabel do dataset.

O resultado foi:

- Teste: 167 imagens
- Desenvolvimento: 671 imagens
- Total: 838 imagens
- Sobreposição entre conjuntos: 0

Entretanto, durante a inspeção dos arquivos RGB foi observado que muitas imagens consecutivas correspondem a regiões próximas da mesma fachada, capturadas durante o deslocamento do drone.

Isso cria risco de imagens visualmente muito semelhantes serem colocadas aleatoriamente em desenvolvimento e teste, causando vazamento de informação e tornando a avaliação excessivamente otimista.

Por esse motivo, a divisão aleatória foi descartada.

---

## 9. Divisão contínua adotada

Para reduzir o problema de imagens consecutivas distribuídas entre os dois conjuntos, foi adotada uma divisão contínua dentro de cada combinação de classes.

Para cada combinação:

1. As imagens são ordenadas pelo nome do arquivo, que acompanha a sequência de captura.
2. Aproximadamente os primeiros 80% são destinados ao desenvolvimento.
3. Aproximadamente os últimos 20% são destinados ao teste.
4. Combinações cujo cálculo resulta em zero imagens de teste permanecem integralmente no desenvolvimento.

Dessa forma:

`[ 80% iniciais: DESENVOLVIMENTO ][ 20% finais: TESTE ]`

O objetivo é reduzir a mistura aleatória de imagens consecutivas entre desenvolvimento e teste.

---

## 10. Resultado da divisão contínua

O resultado final foi:

- Desenvolvimento: 671 imagens (80,07%)
- Teste: 167 imagens (19,93%)
- Total: 838 imagens
- Sobreposição: 0 imagens

Distribuição das classes no conjunto de teste:

| Classe | Total no dataset | Teste | Percentual no teste |
|---|---:|---:|---:|
| CRACK | 833 | 167 | 20,05% |
| PEELING | 188 | 37 | 19,68% |
| HOLLOW_AREA | 383 | 77 | 20,10% |
| EROSION | 584 | 117 | 20,03% |
| STAIN | 472 | 94 | 19,92% |

Portanto, mesmo substituindo a seleção aleatória pela seleção contínua, a distribuição das classes permaneceu muito próxima dos 20% desejados.

---

## 11. Inspeção visual dos pontos de corte

Os pontos de transição entre desenvolvimento e teste foram inspecionados visualmente.

Foram encontrados alguns casos em que as imagens nos dois lados do corte possuem alta similaridade.

### Casos observados

`DJI_20250624190637_0251` / `DJI_20250624190642_0252`

Imagens de regiões lado a lado, visualmente muito semelhantes.

`DJI_20250626185532_0061` / `DJI_20250626185547_0065`

Mesma fachada em regiões diferentes, contendo elementos semelhantes, embora não sejam imagens equivalentes.

`DJI_20250626185225_0030` / `DJI_20250626185227_0031`

Regiões diferentes de uma mesma parede, apresentando o mesmo padrão visual e padrões semelhantes de danos.

`DJI_20250625173514_0039` / `DJI_20250625174210_0069`

Paredes diferentes, porém ambas lisas e contendo fissuras. Uma delas também apresenta áreas ocas.

`DJI_20250626190103_0095` / `DJI_20250626190107_0096`

Duas imagens da mesma fachada, cada uma contendo aproximadamente metade da mesma janela.

`DJI_20250628174529_0015` / `DJI_20250628174533_0016`

Imagens complementares contendo os mesmos elementos em alturas diferentes.

Os demais pontos de corte inspecionados não apresentaram similaridade visual considerada relevante.

---

## 12. Limitações identificadas

A estratégia contínua reduz o risco de vazamento em comparação à divisão aleatória, mas não o elimina completamente.

Como as imagens são agrupadas primeiro pela combinação de classes, imagens consecutivas da captura podem pertencer a combinações diferentes e, portanto, serem separadas antes da realização do corte.

Além disso, algumas sequências do drone percorrem continuamente uma mesma fachada. Assim, mesmo imagens não consecutivas podem compartilhar características visuais.

Uma separação completamente independente exigiria informações adicionais sobre cenas, fachadas ou sequências de captura, que não estão explicitamente disponíveis na estrutura atual do dataset.

Para o escopo deste projeto, foi mantida a divisão contínua e sua limitação deve ser considerada na interpretação dos resultados.

---

## 13. Decisão atual

A divisão adotada para os experimentos é:

- 671 imagens para desenvolvimento;
- 167 imagens para teste;
- divisão aproximada 80/20;
- estratificação pelas combinações de classes;
- ordenação pelo identificador de captura;
- seleção contínua em vez de aleatória;
- nenhuma sobreposição direta de arquivos entre os conjuntos.

O conjunto de desenvolvimento será utilizado durante a criação e ajuste dos algoritmos de processamento.

O conjunto de teste deverá permanecer reservado para a avaliação final, evitando que seus resultados sejam utilizados para ajustar parâmetros do algoritmo.