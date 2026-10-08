# PI-Seg-Fissuras

Projeto da disciplina de **Processamento de Imagens** (UFMA), dedicado à identificação e segmentação de manifestações patológicas em imagens de fachadas por meio de técnicas de Processamento Digital de Imagens.

## Objetivo

Desenvolver um pipeline que receba uma imagem de fachada, aplique pré-processamento e produza máscaras de segmentação para identificar regiões danificadas. A implementação começará pela **detecção de fissuras** e poderá ser ampliada para outras classes.

| ID | Classe (BFDD) | Descrição |
|---:|---|---|
| 0 | `BACKGROUND` | Fundo / regiões não rotuladas como dano |
| 1 | `CRACK` | Fissura |
| 2 | `PEELING` | Descascamento |
| 3 | `HOLLOW_AREA` | Área oca |
| 4 | `EROSION` | Erosão |
| 5 | `STAIN` | Mancha |

O sistema deverá funcionar também com **imagens externas**, sem depender das máscaras do dataset. As máscaras de referência serão usadas **somente na avaliação**, nunca como entrada da segmentação.

## Dataset

Utilizamos o **BFDD (Building Façade Defect Dataset)**, com **838 amostras** e imagens de **512 × 640 pixels**. As pastas disponíveis incluem:

- `RGB/`: fotografias utilizadas como entrada do pipeline;
- `Label/`: máscaras numéricas de referência (IDs de 0 a 5);
- `Label_color/`: representação colorida das classes;
- `IR/`: imagens infravermelhas;
- `Label_backup_7classes_20260125/`: cópia alternativa das labels.

A classe `CRACK` está presente em **833 das 838 imagens**. Como uma amostra pode conter vários danos, as classes não são mutuamente exclusivas no nível da imagem.

### Divisão dos dados — concluída

Para desenvolvimento e avaliação, as imagens foram agrupadas pela **combinação de classes presentes** (desconsiderando o fundo), ordenadas pelo identificador de captura e divididas de forma **contínua** em aproximadamente 80% para desenvolvimento e 20% para teste.

| Conjunto | Imagens | Percentual |
|---|---:|---:|
| Desenvolvimento | 671 | 80,07% |
| Teste | 167 | 19,93% |
| **Total** | **838** | **100%** |

As verificações automatizadas confirmaram **ausência de arquivos repetidos, ausentes e compartilhados entre os conjuntos** (`overlap = 0`). A inspeção visual identificou alguns pares semelhantes nos pontos de corte; portanto, a divisão reduz, mas **não elimina**, o risco de similaridade visual entre desenvolvimento e teste.

O histórico da exploração do BFDD, das combinações e dos testes está documentado em [`DATASET_ANALYSIS.md`](DATASET_ANALYSIS.md).

## Estrutura do projeto

```text
PI-Seg-Fissuras/
├── data/
│   └── BFDD/
│       ├── RGB/
│       ├── IR/
│       ├── Label/
│       ├── Label_color/
│       └── Label_backup_7classes_20260125/
├── src/
│   ├── dataset/
│   │   ├── classes.py          # Enumeração das classes
│   │   ├── loader.py           # Leitura de imagens e labels
│   │   ├── masks.py            # Extração de máscaras binárias
│   │   ├── index.py            # Índice de classes por imagem
│   │   └── split.py            # Divisão desenvolvimento/teste
│   ├── inspect_dataset.py     # Estatísticas e inspeção do BFDD
│   ├── visualize_sample.py    # Inspeção visual de amostras
│   └── test_dataset.py        # Verificações da preparação dos dados
├── requirements.txt
├── DATASET_ANALYSIS.md
└── README.md
```

> A pasta `data/BFDD/` representa a organização local do dataset. Os diretórios de implementação abaixo serão adicionados conforme o desenvolvimento avançar.

### Módulos previstos

```text
src/
├── preprocessing/   # Escala de cinza, contraste e redução de ruído
├── segmentation/    # Algoritmos de segmentação por tipo de dano
├── postprocessing/  # Refinamento das máscaras previstas
├── evaluation/      # IoU, Dice/F1, precisão e recall
└── visualization/   # Comparações e sobreposições dos resultados
results/             # Saídas experimentais e visualizações
```

## Pipeline planejado

```text
Imagem RGB (BFDD ou foto externa)
           ↓
Pré-processamento (grayscale, CLAHE, filtros — a avaliar)
           ↓
Segmentação (inicialmente fissuras)
           ↓
Pós-processamento (morfologia e limpeza — a avaliar)
           ↓
Máscara binária prevista / visualização
           ↓
Avaliação com label de referência (somente para imagens rotuladas)
```

As técnicas específicas e seus parâmetros serão definidos e comparados experimentalmente; **ainda não há resultados de segmentação ou métricas finais**.

## Estado atual e próximas etapas

**Concluído:** inspeção da estrutura e das classes do BFDD; carregamento das imagens e labels; geração de máscaras binárias; indexação; análise das combinações de danos; divisão 80/20 e verificações de integridade.

**A desenvolver:**

1. Implementar e testar o pré-processamento das imagens.
2. Desenvolver a segmentação de fissuras e, posteriormente, investigar outras classes.
3. Refinar as máscaras geradas com pós-processamento.
4. Comparar as previsões com as labels do BFDD usando IoU, Dice/F1, precisão e recall.
5. Gerar visualizações dos resultados e permitir o processamento de fotos externas.

O conjunto de **desenvolvimento** servirá para experimentar métodos e ajustar parâmetros. O conjunto de **teste** ficará reservado para a avaliação final.
