# PI-Seg-Fissuras

Projeto desenvolvido para a disciplina de **Processamento de Imagens**, com o objetivo de identificar e segmentar diferentes tipos de manifestações patológicas presentes em imagens de fachadas.

## Objetivo

Desenvolver um pipeline de processamento e segmentação capaz de analisar imagens de fachadas e identificar regiões correspondentes às seguintes classes:

| ID | Classe       |
|---|---            |
| 0 | Background    |
| 1 | Fissura       |
| 2 | Descascamento |
| 3 | Área oca      |
| 4 | Erosão        |
| 5 | Mancha        |

O projeto utiliza um **dataset público previamente rotulado**, contendo imagens das fachadas e suas respectivas máscaras de segmentação.

## Estrutura atual

```text
PI-Seg-Fissuras/
│
├── data/
│   ├── images/              # Imagens originais
│   └── masks/               # Máscaras de segmentação
│
├── src/
│   └── visualize_sample.py  # Visualização das imagens e labels
│   └── inspect_sataset.py
│
├── requirements.txt
└── README.md
```

Atualmente, já realizamos a organização inicial do dataset e desenvolvemos uma interface de visualização que permite comparar uma imagem original com sua respectiva máscara e identificar visualmente as classes presentes.

## Estrutura prevista

```text
PI-Seg-Fissuras/
│
├── data/
│   ├── images/
│   └── masks/
│
├── src/
│   ├── preprocessing/       # Pré-processamento das imagens
│   ├── segmentation/        # Algoritmos de segmentação
│   ├── evaluation/          # Métricas e comparação dos resultados
│   └── visualization/       # Visualização dos resultados
│
├── results/                 # Imagens, gráficos e resultados
├── requirements.txt
└── README.md
```

## Próximas etapas

O desenvolvimento previsto consiste em:

1. Explorar e compreender melhor o dataset e suas classes;
2. Realizar o pré-processamento das imagens;
3. Implementar técnicas de segmentação e processamento de imagens;
4. Comparar as segmentações obtidas com as máscaras originais do dataset;
5. Avaliar quantitativamente os resultados utilizando métricas de segmentação;
6. Gerar visualizações para análise dos resultados.

O objetivo final é verificar **até que ponto técnicas de Processamento Digital de Imagens conseguem identificar e separar automaticamente as diferentes patologias presentes nas fachadas**, utilizando as máscaras do dataset como referência para avaliação.