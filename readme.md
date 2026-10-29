# PI-Seg-Fissuras

**Segmentação de fissuras e outros danos em fachadas com processamento digital de imagens e aprendizado de máquina.**

Projeto desenvolvido na disciplina de **Processamento de Imagens**, com o objetivo principal de identificar fissuras (*Crack*) em imagens de fachadas. A abordagem atual utiliza **características extraídas por métodos clássicos de processamento de imagens** e um classificador **Random Forest** para realizar segmentação semântica por pixel.

> **Estado atual:** pipeline de treinamento, inferência e avaliação quantitativa implementado. Primeiro modelo (*baseline*) treinado e avaliado nas 132 imagens de validação. O desempenho ainda é insatisfatório, especialmente devido ao excesso de falsos positivos.

## 1. Objetivo

Desenvolver e avaliar um sistema capaz de classificar os pixels de imagens de fachadas em seis categorias, com foco na detecção de fissuras.

Nesta etapa, o sistema utiliza as imagens do dataset **BFDD**. O processamento de fotografias externas ao dataset é uma possibilidade para versões futuras.

## 2. Dataset

O **BFDD** contém **838 imagens**, com resolução de **512 × 640 pixels**, organizadas em quatro conjuntos de arquivos correspondentes:

| Diretório | Conteúdo |
| --- | --- |
| `RGB/` | Fotografias coloridas das fachadas |
| `IR/` | Imagens infravermelhas |
| `Label/` | Máscaras com identificadores numéricos das classes |
| `Label_color/` | Representações coloridas das anotações |

**Na implementação atual**, utilizamos as imagens `RGB` e as máscaras `Label`. Os outros diretórios estão disponíveis para exploração, mas não fazem parte das sete características do modelo atual.

### Classes

| ID | Classe | Descrição |
| --- | --- | --- |
| 0 | `BACKGROUND` | Fundo / regiões sem dano anotado |
| 1 | `CRACK` | Fissura |
| 2 | `PEELING` | Descascamento |
| 3 | `HOLLOW_AREA` | Área oca |
| 4 | `EROSION` | Erosão |
| 5 | `STAIN` | Mancha |

As classes são representadas no projeto pelo enum `DamageClass`, em `src/dataset/classes.py`.

### Divisão dos dados

O dataset foi separado considerando as **combinações de classes presentes nas imagens**:

| Conjunto | Imagens | Uso |
| --- | ---: | --- |
| Treinamento | 539 | Ajuste do classificador |
| Validação | 132 | Análise e comparação de experimentos |
| Teste | 167 | Avaliação final, ainda reservada |
| **Total** | **838** | |

Primeiro, `split_dataset()` separa desenvolvimento (671 imagens) e teste (167). Em seguida, `split_train_validation()` divide o desenvolvimento em treinamento e validação, com semente aleatória `42`.

Não há sobreposição de nomes de imagens entre os conjuntos. **A divisão por nomes não garante, por si só, independência visual entre fotografias próximas de uma mesma fachada**, aspecto a considerar na interpretação dos resultados.

## 3. Pipeline implementado

```text
Imagem RGB + máscara de referência (somente treinamento/avaliação)
       │
       ├── Imagem RGB
       │      ↓
       │   Conversão para tons de cinza (luminosidade)
       │      ↓
       │   Extração de 7 características por pixel
       │      ↓
       │   ┌─────────────────────┬──────────────────────┐
       │   │ TREINAMENTO         │ PREDIÇÃO             │
       │   │ Amostragem por      │ Todos os pixels      │
       │   │ classe + labels     │ sem usar a label     │
       │   │        ↓            │         ↓            │
       │   │ Random Forest.fit() │ Random Forest.predict()
       │   └─────────────────────┴──────────────────────┘
       │                                  ↓
       │                         Máscara prevista (512 × 640)
       │                                  ↓
       └── Label real ─────────→ Avaliação por pixel
                                  (TP, FP, FN, Precision,
                                   Recall, F1-score e IoU)
```

### 3.1. Pré-processamento

- Leitura da imagem RGB.
- Conversão para escala de cinza por luminosidade.
- Preservação das dimensões espaciais da imagem.

Outras técnicas de melhoramento, como **CLAHE**, podem ser investigadas em experimentos futuros, mas **não integram o baseline atual**.

### 3.2. Extração de características

Para cada pixel, `extract_features(image, gray)` produz um vetor com **7 características**:

1. Canal vermelho (`R`);
2. Canal verde (`G`);
3. Canal azul (`B`);
4. Intensidade em escala de cinza;
5. Média local em janela `3 × 3`;
6. Desvio-padrão local em janela `3 × 3`;
7. Magnitude do gradiente de Sobel.

Para uma imagem de 512 × 640 pixels, a matriz de características tem formato **`(327680, 7)`**, em `float32`.

### 3.3. Amostragem para treinamento

A função `sample_pixels()` seleciona aleatoriamente **até 500 pixels por classe em cada imagem**, conforme sua disponibilidade na anotação. A amostragem é reproduzível por meio de gerador aleatório com semente definida.

O conjunto amostrado de treinamento possui:

- `X_train`: **`(995106, 7)`**
- `y_train`: **`(995106,)`**

Distribuição das amostras:

| Classe | Pixels amostrados |
| --- | ---: |
| Background | 269.500 |
| Crack | 262.976 |
| Peeling | 52.561 |
| Hollow Area | 112.363 |
| Erosion | 153.614 |
| Stain | 144.092 |

Essa estratégia diminui o domínio do background no treinamento, mas pode contribuir para uma distribuição de classes diferente daquela encontrada nas imagens reais. Trata-se de uma **hipótese a investigar**, não de uma causa já comprovada para os erros.

### 3.4. Treinamento

Classificador: `sklearn.ensemble.RandomForestClassifier`.

| Parâmetro | Valor |
| --- | --- |
| `n_estimators` | 100 |
| `criterion` | `gini` |
| `max_depth` | 20 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 5 |
| `max_features` | `sqrt` |
| `class_weight` | `None` |
| `n_jobs` | `-1` |
| `random_state` | 42 |

O modelo treinado é salvo com `joblib` em **`models/random_forest.joblib`** (aproximadamente 1 GiB).

### 3.5. Predição

A função `predict_image(model, image)`:

1. Prepara a imagem e extrai as mesmas sete características usadas no treinamento;
2. Aplica `model.predict()` a **todos os pixels** da imagem, sem amostragem;
3. Reorganiza as previsões em uma máscara **`(512, 640)`** com identificadores de classe de 0 a 5.

A **label real não é utilizada na predição**; ela é usada apenas para treinamento supervisionado ou avaliação.

### 3.6. Avaliação

A avaliação é feita pixel a pixel, considerando cada classe separadamente (*one-vs-rest*):

- **TP:** pixel pertence à classe e foi previsto corretamente;
- **FP:** pixel não pertence à classe, mas foi previsto como pertencente;
- **FN:** pixel pertence à classe, mas foi previsto como outra.

Métricas:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-score  = 2 × Precision × Recall / (Precision + Recall)
IoU       = TP / (TP + FP + FN)
```

Divisões por zero são tratadas no código. Quando uma classe não aparece nem na referência nem na previsão, a métrica é indefinida; esse caso deve ser tratado explicitamente em futuras médias por imagem.

Na avaliação global, **somamos TP, FP e FN das 132 imagens de validação antes de calcular as métricas**. Não utilizamos uma média simples das métricas individuais das imagens como resultado global por pixel.

## 4. Resultados do baseline

**Conjunto avaliado:** 132 imagens de validação, totalizando **43.253.760 pixels**. As 167 imagens de teste não foram utilizadas.

### Contagens acumuladas

| Classe | TP | FP | FN |
| --- | ---: | ---: | ---: |
| Background | 26.950.027 | 977.816 | 12.350.159 |
| Crack | 491.519 | 5.540.731 | 245.773 |
| Peeling | 53.318 | 566.929 | 236.266 |
| Hollow Area | 117.368 | 2.911.036 | 314.226 |
| Erosion | 744.405 | 2.778.177 | 495.125 |
| Stain | 245.805 | 1.876.629 | 1.009.769 |

### Métricas globais por classe

| Classe | Precision | Recall | F1-score | IoU |
| --- | ---: | ---: | ---: | ---: |
| Background | 96,50% | 68,57% | 80,17% | 66,91% |
| **Crack** | **8,15%** | **66,67%** | **14,52%** | **7,83%** |
| Peeling | 8,60% | 18,41% | 11,72% | 6,23% |
| Hollow Area | 3,88% | 27,19% | 6,78% | 3,51% |
| Erosion | 21,13% | 60,06% | 31,26% | 18,53% |
| Stain | 11,58% | 19,58% | 14,55% | 7,85% |

> **Interpretação:** para `CRACK`, o recall de aproximadamente 66,7% indica que o modelo encontra cerca de dois terços dos pixels reais de fissura, mas a precisão de 8,15% evidencia um número elevado de falsos positivos. A IoU de 7,83% mostra que a segmentação ainda precisa melhorar substancialmente.

Esses resultados constituem o **baseline** para comparar novas configurações de pré-processamento, características, amostragem e treinamento.

## 5. Estrutura do projeto

Estrutura conhecida até o momento, com as pastas de dados e scripts de execução/teste:

```text
PI-Seg-Fissuras/
├── data/
│   ├── BFDD/
│   │   ├── RGB/
│   │   ├── IR/
│   │   ├── Label/
│   │   └── Label_color/
│   └── models/                  # Diretório auxiliar, se mantido
├── models/
│   └── random_forest.joblib     # Modelo treinado
├── src/
│   ├── dataset/
│   │   ├── classes.py           # Enum das classes de dano
│   │   ├── index.py             # Índice e classes presentes por imagem
│   │   ├── loader.py            # Leitura de RGB e labels
│   │   ├── masks.py             # Máscaras binárias por classe
│   │   └── split.py             # Divisão treino/validação/teste
│   ├── tests/
│   │   ├── test_dataset.py      # Verificações do dataset
│   │   ├── test_features.py     # Verificações das características
│   │   ├── test_training.py     # Rotina de teste/treinamento
│   │   └── test_prediction.py   # Predição e validação
│   ├── features.py              # Características e amostragem de pixels
│   ├── preprocessing.py         # Conversão para cinza e pré-processamento
│   ├── training.py              # Montagem de dados e Random Forest
│   ├── prediction.py            # Inferência por pixel
│   ├── evaluation.py            # TP, FP, FN e métricas
│   ├── inspect_dataset.py       # Inspeção de arquivos e dimensões
│   └── visualize_sample.py      # Visualização de imagens e anotações
├── .gitignore
├── DATASET_ANALYSIS.md
├── requirements.txt
└── README.md
```

**Observação:** a árvore reflete os arquivos discutidos e implementados até agora; confirme eventuais nomes adicionais ou diferenças na sua cópia local.

### Responsabilidades dos principais módulos

| Módulo | Responsabilidade |
| --- | --- |
| `dataset/index.py` | Identificar classes presentes em cada imagem |
| `dataset/split.py` | Gerar conjuntos sem sobreposição |
| `features.py` | Extrair sete características e amostrar pixels |
| `training.py` | Preparar dados de treino e ajustar o classificador |
| `prediction.py` | Gerar máscara multiclasse para uma imagem |
| `evaluation.py` | Calcular contagens e métricas por classe |
| `tests/` | Executar verificações e experimentos do pipeline |

## 6. Ambiente e execução

**Tecnologias:** Python 3, NumPy, OpenCV, scikit-learn, joblib, Pillow e Matplotlib (conforme utilização nos scripts).

No diretório raiz do projeto, instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Ajuste o caminho dos dados para corresponder à estrutura acima. Os scripts de teste importam módulos diretamente de `src`, portanto é necessário incluí-lo no caminho de importação.

**PowerShell (Windows):**

```powershell
$env:PYTHONPATH="src"
python src/tests/test_prediction.py
```

**Linux/macOS:**

```bash
PYTHONPATH=src python src/tests/test_prediction.py
```

A execução de predição pressupõe que `models/random_forest.joblib` já exista. **Não é necessário treinar novamente** para repetir a avaliação de validação.

Os arquivos do dataset e o modelo serializado são volumosos e devem ser mantidos fora do controle de versão, conforme o `.gitignore`.

## 7. Etapas concluídas e próximas atividades

### Concluído

- [x] Inspeção do BFDD e identificação das seis classes.
- [x] Carregamento das imagens e máscaras.
- [x] Separação dos conjuntos de treinamento, validação e teste.
- [x] Conversão para tons de cinza.
- [x] Extração de sete características por pixel.
- [x] Amostragem de pixels por classe.
- [x] Treinamento e persistência da Random Forest.
- [x] Predição multiclasse de imagens completas.
- [x] Cálculo de TP, FP, FN, Precision, Recall, F1-score e IoU.
- [x] Avaliação acumulada das 132 imagens de validação.

### Próximas etapas

- [ ] Visualizar lado a lado a imagem RGB, a label real e a máscara prevista.
- [ ] Investigar visualmente os falsos positivos, sobretudo da classe `CRACK`.
- [ ] Testar outras estratégias de amostragem e representação do background.
- [ ] Comparar novos experimentos com o baseline, mantendo o mesmo conjunto de validação.
- [ ] Investigar melhorias nas características e no pré-processamento.
- [ ] Selecionar uma configuração final e avaliá-la **uma única vez** no conjunto de teste reservado.
- [ ] Documentar os resultados finais e as limitações do método.
- [ ] Considerar, futuramente, a predição em fotografias externas ao BFDD.

## 8. Limitações atuais

O modelo apresenta excesso de falsos positivos e baixa IoU para fissuras. As causas ainda precisam ser investigadas experimentalmente. Entre as hipóteses estão o desequilíbrio entre a distribuição de pixels amostrados e a distribuição real, a capacidade discriminativa das sete características e a semelhança visual entre defeitos e elementos do fundo.

A avaliação atual é **por pixel**; não mede diretamente a qualidade de fissuras como objetos contínuos, sua conectividade ou localização geométrica. Também não demonstra generalização para fachadas fora do dataset.

---

**Status do projeto:** baseline concluído; em fase de análise de erros e melhoria da segmentação.
