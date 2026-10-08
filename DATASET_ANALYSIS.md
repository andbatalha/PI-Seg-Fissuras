# BFDD — Registro de análise, preparação e validação do dataset

**Projeto:** Segmentação de fissuras em fachadas — Processamento de Imagens (UFMA)  
**Dataset:** BFDD (*Building Façade Defect Dataset*)  
**Última atualização:** 07/10/2026  
**Estado:** Análise do dataset e implementação do split concluídas; pré-processamento e segmentação ainda não iniciados.

## 1. Objetivo deste registro

Documentar as inspeções, testes, experimentos e decisões tomados **antes da implementação do pipeline de processamento e segmentação**. O registro diferencia resultados efetivamente verificados de limitações ainda não resolvidas, para que as escolhas possam ser reproduzidas e justificadas no relatório do trabalho.

O objetivo do sistema é processar uma imagem de fachada, identificar regiões de fissuras e produzir uma máscara de segmentação. As máscaras fornecidas pelo BFDD serão utilizadas como **referência para avaliação**, não como entrada do algoritmo de segmentação. Futuramente, o pipeline deverá aceitar também fotografias externas ao BFDD, que não necessariamente terão labels.

## 2. Estrutura e características do BFDD

Na estrutura inspecionada, foram identificadas cinco pastas com **838 arquivos cada**:

| Pasta | Conteúdo | Quantidade |
|---|---|---:|
| `RGB/` | Fotografias coloridas das fachadas | 838 |
| `IR/` | Imagens infravermelhas | 838 |
| `Label/` | Máscaras numéricas das classes | 838 |
| `Label_color/` | Representações coloridas das labels | 838 |
| `Label_backup_7classes_20260125/` | Cópia alternativa das labels presente no dataset | 838 |
| **Total** | **Arquivos nas cinco pastas** | **4.190** |

Os 4.190 arquivos correspondem a **838 amostras identificadas por nomes de arquivo**, não a 4.190 fotografias independentes.

Dimensões e formatos observados:

- RGB: `512 × 640 × 3` (altura × largura × canais).
- IR: `512 × 640 × 3` na estrutura inspecionada.
- Label numérica: `512 × 640`, com valores inteiros do tipo `uint8`.
- Label colorida: `512 × 640 × 3`.

**Decisão:** utilizar `RGB/` como entrada do processamento e `Label/` como verdade de referência. `Label_color/` auxilia na inspeção visual. Não foi definido uso das imagens IR ou das labels de backup no pipeline atual.

## 3. Classes presentes nas labels

A inspeção dos valores únicos das máscaras identificou os IDs **0 a 5**. O mapeamento adotado no projeto é:

| ID | Nome no código | Interpretação |
|---:|---|---|
| 0 | `BACKGROUND` | Fundo / região sem uma das manifestações rotuladas |
| 1 | `CRACK` | Fissura |
| 2 | `PEELING` | Descascamento |
| 3 | `HOLLOW_AREA` | Área oca |
| 4 | `EROSION` | Erosão |
| 5 | `STAIN` | Mancha |

Esse mapeamento foi centralizado em `src/dataset/classes.py`, por meio de `DamageClass(IntEnum)`.

Na inspeção visual, as classes `BACKGROUND`, `CRACK` e `HOLLOW_AREA` foram mais fáceis de reconhecer. A distinção entre `PEELING`, `EROSION` e `STAIN` apresentou ambiguidades visuais em algumas amostras. **Os rótulos originais não foram modificados.**

## 4. Teste do carregamento de imagens e labels

Foram implementadas funções reutilizáveis em `src/dataset/loader.py`:

- `load_image(image_path)`: lê a imagem e converte a ordem de canais do OpenCV de BGR para RGB.
- `load_label(label_path)`: lê a máscara numérica em escala de cinza, preservando os IDs de classe.

Verificações em amostras:

- Imagem RGB carregada: forma `(512, 640, 3)`.
- Label carregada: forma `(512, 640)`.
- Valores únicos de uma label testada: `[0, 1, 2, 5]`.

O teste confirmou que uma mesma imagem pode conter mais de uma manifestação patológica.

## 5. Teste da criação de máscaras binárias

Foi implementada em `src/dataset/masks.py` a função:

```python
create_binary_mask(label, damage_class)
```

Ela compara os pixels da label ao ID da classe solicitada e retorna uma máscara binária `uint8`:

- `1`: pixel pertence à classe solicitada;
- `0`: pixel não pertence à classe solicitada.

**Resultados em uma amostra:**

- Máscara de `CRACK`: **620 pixels positivos**.
- Máscara de `EROSION` da mesma amostra: **0 pixels positivos**, pois a classe não estava presente naquela label.

Esse teste verificou a extração de máscaras específicas a partir de uma label multiclasses. **Não foi um teste de detecção de fissuras pelo algoritmo**, que ainda não havia sido implementado.

## 6. Indexação das classes presentes por imagem

Foram criadas em `src/dataset/index.py` as funções:

- `get_present_classes(label)`: identifica as classes presentes nos valores únicos da máscara.
- `build_dataset_index(labels_dir)`: percorre as labels e associa cada nome de imagem à lista de classes encontradas.

Estrutura conceitual:

```python
{
    "imagem_0001": [DamageClass.BACKGROUND, DamageClass.CRACK],
    "imagem_0002": [DamageClass.BACKGROUND, DamageClass.CRACK, DamageClass.STAIN],
}
```

**Resultado:** o índice foi construído com **838 entradas**.

## 7. Frequência das classes no dataset

Contou-se em quantas imagens aparece **ao menos um pixel** de cada classe:

| Classe | Imagens com a classe | Proporção do dataset |
|---|---:|---:|
| `BACKGROUND` | 838 | 100,0% |
| `CRACK` | 833 | 99,4% |
| `PEELING` | 188 | 22,4% |
| `HOLLOW_AREA` | 383 | 45,7% |
| `EROSION` | 584 | 69,7% |
| `STAIN` | 472 | 56,3% |

**Atenção:** são frequências por **imagem**, não proporções de pixels ou de área ocupada por cada dano. Como as classes coexistem, as contagens não devem ser somadas para obter 838.

O fato de `CRACK` aparecer em **833 das 838 imagens** influenciou a escolha da estratégia de divisão do dataset: estratificar apenas pela presença de fissura seria pouco informativo.

## 8. Agrupamento pelas combinações de manifestações

Para cada imagem, `BACKGROUND` foi removido da lista de classes e as classes restantes foram convertidas em uma tupla. Imagens com a mesma combinação exata foram reunidas em `combination_groups`.

Foram identificadas **19 combinações**:

| Combinação de danos | Imagens |
|---|---:|
| `CRACK + PEELING + STAIN` | 19 |
| `CRACK + STAIN` | 92 |
| `CRACK + EROSION + STAIN` | 133 |
| `SEM DANOS` | 2 |
| `CRACK` | 68 |
| `CRACK + HOLLOW_AREA + EROSION` | 109 |
| `CRACK + HOLLOW_AREA + EROSION + STAIN` | 128 |
| `CRACK + PEELING + HOLLOW_AREA + EROSION + STAIN` | 56 |
| `CRACK + PEELING` | 20 |
| `PEELING` | 2 |
| `CRACK + HOLLOW_AREA` | 26 |
| `CRACK + HOLLOW_AREA + STAIN` | 12 |
| `CRACK + EROSION` | 79 |
| `CRACK + PEELING + EROSION + STAIN` | 27 |
| `CRACK + PEELING + EROSION` | 12 |
| `CRACK + PEELING + HOLLOW_AREA + EROSION` | 39 |
| `CRACK + PEELING + HOLLOW_AREA` | 8 |
| `CRACK + PEELING + HOLLOW_AREA + STAIN` | 5 |
| `EROSION` | 1 |
| **Total** | **838** |

Verificação de consistência: as **cinco imagens sem `CRACK`** correspondem exatamente a duas sem danos, duas com apenas `PEELING` e uma com apenas `EROSION`.

## 9. Definição dos conjuntos de desenvolvimento e teste

Como o projeto utiliza inicialmente **processamento clássico de imagens**, e não o treinamento de um modelo supervisionado, decidiu-se utilizar dois conjuntos:

- **Desenvolvimento:** imagens utilizadas para experimentar técnicas e ajustar parâmetros do pipeline.
- **Teste:** imagens reservadas para avaliar o resultado final, sem orientar o ajuste dos parâmetros.

Meta aproximada: **80% desenvolvimento / 20% teste**, preservando a distribuição das combinações de classes.

## 10. Primeiro experimento: divisão aleatória por combinação

O primeiro split embaralhava as imagens **dentro de cada combinação**, usando `random.shuffle` e semente `42`, e reservava aproximadamente 20% de cada grupo para teste.

Resultados obtidos:

- Desenvolvimento: **671** imagens.
- Teste: **167** imagens.
- Sobreposição direta de nomes: **0**.
- Distribuição de classes próxima de 20%.

**Problema identificado:** na visualização dos arquivos RGB, várias fotos consecutivas do drone mostravam regiões sobrepostas ou muito próximas da mesma fachada. O embaralhamento poderia colocar imagens quase equivalentes em desenvolvimento e teste, aumentando o risco de uma avaliação otimista.

**Decisão:** abandonar a seleção aleatória, mesmo com a distribuição de classes satisfatória.

Também foi discutida uma alternativa com 75% iniciais, 10% intermediários embaralhados e 15% finais. Ela **não foi implementada**, pois misturar a faixa intermediária poderia criar novas fronteiras entre imagens semelhantes.

## 11. Estratégia adotada: divisão contínua por combinação

Para cada uma das 19 combinações:

1. Ordenar os nomes das imagens com `sorted(image_names)`.
2. Calcular `test_count = round(len(image_names) * 0.2)`.
3. Se `test_count == 0`, colocar todas as imagens do grupo em desenvolvimento.
4. Caso contrário, reservar as **últimas `test_count` imagens** para teste e as anteriores para desenvolvimento.

Representação:

```text
Imagens da mesma combinação, ordenadas pelo nome/captura
[        aproximadamente 80% DEV        |   20% TESTE   ]
```

A ordenação lexical dos identificadores DJI acompanha aproximadamente a ordem de captura. A estratégia é **determinística** e não requer `random.seed`.

Casos pequenos:

| Combinação | Total | Desenvolvimento | Teste |
|---|---:|---:|---:|
| `SEM DANOS` | 2 | 2 | 0 |
| `PEELING` apenas | 2 | 2 | 0 |
| `EROSION` apenas | 1 | 1 | 0 |

Esses grupos não foram forçados a produzir uma amostra de teste, para preservar a regra objetiva de divisão.

## 12. Resultado final do split contínuo

| Conjunto | Imagens | Percentual |
|---|---:|---:|
| Desenvolvimento | **671** | **80,07%** |
| Teste | **167** | **19,93%** |
| **Total** | **838** | **100%** |

Distribuição por classe no teste:

| Classe | Total no dataset | Imagens no teste | Fração da classe destinada ao teste |
|---|---:|---:|---:|
| `CRACK` | 833 | 167 | 20,05% |
| `PEELING` | 188 | 37 | 19,68% |
| `HOLLOW_AREA` | 383 | 77 | 20,10% |
| `EROSION` | 584 | 117 | 20,03% |
| `STAIN` | 472 | 94 | 19,92% |

**Conclusão:** a divisão contínua manteve uma distribuição de classes muito próxima da obtida na divisão aleatória, reduzindo a mistura de imagens consecutivas **dentro de cada combinação**.

Distribuição final **por combinação**, reproduzida na execução do teste:

| Combinação | Total | Desenvolvimento | Teste |
|---|---:|---:|---:|
| `CRACK + PEELING + STAIN` | 19 | 15 | 4 |
| `CRACK + STAIN` | 92 | 74 | 18 |
| `CRACK + EROSION + STAIN` | 133 | 106 | 27 |
| `SEM DANOS` | 2 | 2 | 0 |
| `CRACK` | 68 | 54 | 14 |
| `CRACK + HOLLOW_AREA + EROSION` | 109 | 87 | 22 |
| `CRACK + HOLLOW_AREA + EROSION + STAIN` | 128 | 102 | 26 |
| `CRACK + PEELING + HOLLOW_AREA + EROSION + STAIN` | 56 | 45 | 11 |
| `CRACK + PEELING` | 20 | 16 | 4 |
| `PEELING` | 2 | 2 | 0 |
| `CRACK + HOLLOW_AREA` | 26 | 21 | 5 |
| `CRACK + HOLLOW_AREA + STAIN` | 12 | 10 | 2 |
| `CRACK + EROSION` | 79 | 63 | 16 |
| `CRACK + PEELING + EROSION + STAIN` | 27 | 22 | 5 |
| `CRACK + PEELING + EROSION` | 12 | 10 | 2 |
| `CRACK + PEELING + HOLLOW_AREA + EROSION` | 39 | 31 | 8 |
| `CRACK + PEELING + HOLLOW_AREA` | 8 | 6 | 2 |
| `CRACK + PEELING + HOLLOW_AREA + STAIN` | 5 | 4 | 1 |
| `EROSION` | 1 | 1 | 0 |
| **Total** | **838** | **671** | **167** |

## 13. Inspeção visual dos pontos de corte

Foram identificados **16 grupos com imagens nos dois conjuntos** (19 combinações, das quais 3 têm zero imagens de teste). As imagens imediatamente antes e depois de cada corte foram inspecionadas visualmente.

Em **6 cortes**, observou-se alguma similaridade visual relevante:

| Última imagem de desenvolvimento | Primeira imagem de teste | Observação visual |
|---|---|---|
| `DJI_20250624190637_0251` | `DJI_20250624190642_0252` | Regiões lado a lado, muito parecidas. |
| `DJI_20250626185532_0061` | `DJI_20250626185547_0065` | Mesma fachada, locais distintos, elementos semelhantes. |
| `DJI_20250626185225_0030` | `DJI_20250626185227_0031` | Regiões diferentes da mesma parede, com padrão visual e de danos semelhante. |
| `DJI_20250625173514_0039` | `DJI_20250625174210_0069` | Paredes diferentes, ambas lisas e fissuradas; uma possui áreas ocas. |
| `DJI_20250626190103_0095` | `DJI_20250626190107_0096` | Fotografias complementares, cada uma mostrando parte da mesma janela. |
| `DJI_20250628174529_0015` | `DJI_20250628174533_0016` | Fotografias complementares dos mesmos elementos em alturas diferentes. |

Nos demais cortes inspecionados, não foi observada similaridade considerada relevante.

**Decisão:** manter os cortes definidos pela regra contínua, sem deslocamentos manuais, para evitar decisões subjetivas de seleção de imagens. A inspeção documenta uma limitação, mas **não comprova independência visual** entre desenvolvimento e teste.

## 14. Refatoração: implementação definitiva em `split.py`

Após os experimentos em `src/test_dataset.py`, a lógica de divisão foi transferida para:

```text
src/dataset/split.py
```

A função reutilizável é:

```python
split_dataset(dataset_index)
```

**Entrada:** dicionário `dataset_index` com o nome de cada imagem e suas classes.  
**Saída:** tupla contendo `(development_images, test_images)`, duas listas de identificadores de imagens.

Responsabilidades implementadas:

1. Percorrer o índice e remover `DamageClass.BACKGROUND` das combinações.
2. Agrupar os identificadores por tuplas de classes presentes.
3. Ordenar os identificadores dentro de cada combinação.
4. Aplicar o corte contínuo aproximado 80/20.
5. Tratar grupos pequenos cujo `test_count` seja zero.
6. Retornar as duas listas, sem depender de prints de diagnóstico.

O arquivo `test_dataset.py` passou a **importar** a função, em vez de conter uma segunda implementação do split:

```python
from dataset.split import split_dataset

dataset_index = build_dataset_index(labels_dir)
development_images, test_images = split_dataset(dataset_index)
```

## 15. Teste de integração e validação de integridade

Após a refatoração, `test_dataset.py` foi executado novamente e produziu:

```text
=== RESULTADO DO SPLIT ===
Teste: 167 imagens
Desenvolvimento: 671 imagens
Total: 838 imagens
Overlap: 0

=== CONTAGEM DE CLASSES PARA TESTE ===
CRACK: 167
PEELING: 37
STAIN: 94
EROSION: 117
HOLLOW_AREA: 77
```

O resultado **reproduziu o experimento anterior**, confirmando que a extração da lógica para `split.py` não alterou as contagens.

Foram acrescentadas as seguintes verificações automáticas:

```python
# Nenhuma duplicação dentro de cada lista.
assert len(development_images) == len(set(development_images))
assert len(test_images) == len(set(test_images))

# União dos conjuntos contém exatamente as imagens indexadas.
all_images = set(development_images) | set(test_images)
assert all_images == set(dataset_index.keys())
```

Também foi verificada a interseção:

```python
overlap = set(test_images) & set(development_images)
print(f"Overlap: {len(overlap)}")  # 0
```

**Resultado observado:** o programa terminou **sem erros de `assert`**.

Isso verifica, em conjunto:

- ausência de identificadores duplicados em cada lista;
- ausência de identificadores simultaneamente em desenvolvimento e teste;
- cobertura de todos os 838 identificadores do índice;
- manutenção das contagens esperadas após a refatoração.

Esses testes verificam a **integridade estrutural da divisão**, não a independência visual ou estatística das fotografias.

## 16. Arquivos do projeto envolvidos até esta etapa

```text
src/
├── dataset/
│   ├── classes.py        # Enumeração das classes
│   ├── loader.py         # Carregamento de RGB e labels
│   ├── masks.py          # Extração de máscara binária por classe
│   ├── index.py          # Classes presentes e índice das imagens
│   └── split.py          # Divisão contínua desenvolvimento/teste
├── inspect_dataset.py   # Inspeção da estrutura, valores e frequências
├── visualize_sample.py  # Inspeção visual de imagens e labels
└── test_dataset.py      # Testes de integração e integridade do dataset
```

O teste é executado a partir da raiz do projeto com:

```bash
python src/test_dataset.py
```

## 17. Limitações e cuidados para o relatório

1. **Similaridade entre fotografias:** a divisão contínua reduz o embaralhamento de fotos consecutivas, mas não elimina cenas semelhantes nos dois conjuntos.
2. **Agrupamento por combinação:** imagens consecutivas com combinações de classes diferentes podem cair em conjuntos distintos, mesmo sem estarem na mesma fronteira de corte inspecionada.
3. **Sem independência garantida por fachada:** não foi feita separação por identidade da fachada, edifício ou sessão de captura. Portanto, não se deve afirmar que o conjunto de teste contém apenas fachadas inéditas.
4. **Ausência de negativos completos no teste:** as 167 imagens de teste contêm `CRACK`; as cinco imagens sem fissuras ficaram em desenvolvimento. Isso limita a avaliação de falsos positivos em **imagens inteiramente sem fissuras**, embora existam pixels negativos nas máscaras de teste.
5. **Classes visualmente ambíguas:** `PEELING`, `EROSION` e `STAIN` podem ser difíceis de diferenciar visualmente em certas imagens. As labels foram mantidas conforme o dataset.
6. **Frequência por imagem, não por pixel:** ainda não foi medida a proporção total de pixels de cada classe, nem a dificuldade das amostras.
7. **Não há resultado de segmentação nesta etapa:** nenhuma métrica como IoU, Dice/F1, precisão ou recall foi calculada para previsões do sistema, pois o algoritmo ainda não foi implementado.

## 18. Próximos passos — fora do escopo deste registro

A etapa de preparação do dataset foi concluída. O desenvolvimento do sistema deverá seguir, inicialmente, esta organização conceitual:

```text
Imagem RGB
    ↓
Pré-processamento (escala de cinza, CLAHE, redução de ruído)
    ↓
Segmentação de fissuras (método a implementar e testar)
    ↓
Pós-processamento da máscara prevista
    ↓
Avaliação contra a máscara binária de CRACK extraída da label
    ↓
Métricas: IoU, Dice/F1, precisão e recall
```

O pipeline de inferência deve funcionar **sem acesso à label**. As labels serão usadas somente na avaliação com o BFDD. Fotografias externas sem anotação poderão ser segmentadas, mas não terão métricas quantitativas de acerto sem uma referência anotada.

---

### Resumo da decisão metodológica

Foi adotada uma divisão **determinística, contínua e estratificada pelas combinações de classes**, com **671 imagens para desenvolvimento** e **167 para teste**. A implementação está isolada em `src/dataset/split.py` e foi testada por meio de `src/test_dataset.py`. Os testes de cobertura, unicidade e interseção passaram. A inspeção visual identificou similaridade em seis fronteiras, registrada como limitação metodológica.
