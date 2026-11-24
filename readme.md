# PI-Seg-Fissuras

Segmentação binária de manchas em fachadas do BFDD. O pipeline segmenta o RGB em superpixels SLIC, calcula 15 características de cor, contraste, textura e gradiente por região e usa um Random Forest para classificar cada região. A máscara final tem valor `1` para Stain (ID 5 no BFDD) e `0` para as demais classes.

## Instalação

Na raiz do repositório, ative a `.venv`, instale as dependências e coloque o BFDD em `data/BFDD/`, com `RGB/` e `Label/`.

```powershell
python -m pip install -r requirements.txt
```

Se a `.venv` foi criada com `uv` sem `pip`, use `uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt`. Para reutilizar um modelo salvo, use o ambiente compatível com o estimador serializado e leia apenas artefatos Joblib confiáveis do projeto.

## Inspeção visual

Com `split.json` e `random_forest.joblib` em `results/region_full/`, execute `python run_inspector.py` na raiz ou abra o arquivo pelo Run do VS Code. Também é possível usar os comandos abaixo.

```powershell
python -m streamlit run visualization/app.py --server.address 127.0.0.1
```

A interface compara RGB, referência binária, predição e mapa de erros. Ela permite navegar por treino e validação. `pixel_validation_per_image.json` habilita a ordenação por métricas. Os NPZ e o relatório por imagem fornecem a presença de Stain para os filtros.

## Construção, treinamento e avaliação

Use os comandos abaixo quando precisar gerar artefatos novos. Para inspecionar um modelo existente, pule a construção e o treinamento. A opção `--output` indica a pasta de trabalho. A interface procura os artefatos em `results/region_full/`.

```powershell
python -m experiments.train_regions build --output results/region_full
python -m experiments.train_regions train --output results/region_full
python -m experiments.evaluate_regions --output results/region_full
```

`build` grava o manifesto, as características e os rótulos em `train.npz` e `validation.npz`, além de `dataset_statistics.json`. Para uma execução reduzida em outra pasta, use `build --smoke --output results/region_smoke`.

`train` lê os NPZ, usa as regiões não ambíguas e salva `random_forest.joblib` e `validation_metrics.json`. Esse relatório avalia a classificação por **região**. `evaluate_regions` usa os IDs de validação do manifesto, acumula contagens por **pixel** e gera `pixel_validation_metrics.json`, `pixel_validation_per_image.json` e exemplos PNG. Para gerar a visualização de um único ID, acrescente `--image-id ID_DE_VALIDACAO`.

A [documentação do BFDD](docs/dataset.md) registra as decisões da divisão e suas limitações. O conjunto de teste permanece reservado.

## Diagnósticos e testes

O diagnóstico do dataset lê pares apenas dos subconjuntos de treino ou validação registrados no manifesto. O diagnóstico SLIC recebe um par RGB e Label explícito. Use caminhos de uma imagem de treino ou validação para examinar fronteiras, categorias supervisionadas e IoU máxima teórica por região.

```powershell
python -m experiments.inspect_dataset --subset validation --limit 5
python -m experiments.inspect_regions --image data/BFDD/RGB/ID.JPG --label data/BFDD/Label/ID.png
python -m unittest discover -s tests -v
```

O PNG do diagnóstico SLIC é gravado em `results/diagnostics/diagnostico_slic.png` por padrão. Use `--output` para indicar outro destino.

## Organização

- `src/` contém o núcleo do pipeline e as métricas compartilhadas.
- `experiments/` contém os comandos de construção, treino, avaliação e diagnóstico.
- `visualization/` contém a interface Streamlit e os auxiliares de figuras.
- `tests/` contém testes automatizados com dados pequenos.
- `results/` contém artefatos locais e é ignorado pelo Git.

A implementação multiclasse anterior por pixel está preservada na branch `main` no ponto histórico `21685cf`.
