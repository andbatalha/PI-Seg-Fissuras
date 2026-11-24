# BFDD e divisão do dataset

O BFDD contém 838 pares RGB e Label de 512 × 640 pixels. A configuração atual usa a máscara numérica `Label/`, com os IDs definidos em `src/dataset/classes.py`. O alvo é `STAIN` (ID 5). As demais classes são negativas na avaliação binária, mas continuam disponíveis para indexar e dividir o dataset. IR e `Label_color/` são fornecidos pelo BFDD e ficam fora da extração atual.

| ID | Classe |
|---:|---|
| 0 | Background |
| 1 | Crack |
| 2 | Peeling |
| 3 | Hollow Area |
| 4 | Erosion |
| 5 | Stain |

## Partição do projeto

1. O índice associa cada ID de imagem à combinação de classes presentes, ignorando `BACKGROUND` ao montar os grupos.
2. Em cada combinação, `split_dataset()` ordena os IDs lexicograficamente e reserva os últimos `round(0.2 * tamanho_do_grupo)` para o teste. Grupos com arredondamento zero ficam no desenvolvimento. São **671 imagens de desenvolvimento** e **167 reservadas**.
3. Dentro do desenvolvimento, `split_train_validation()` embaralha os IDs ordenados de cada grupo com um gerador NumPy e semente `42`. Reserva aproximadamente 20% por grupo para validação. A construção atual ordena o índice por ID antes de chamar as funções, fixando também a ordem dos grupos. A execução registrada tem **539 imagens de treino** e **132 de validação**.

O arquivo local `results/region_full/split.json` registra a divisão usada para gerar os artefatos atuais. Use os IDs desse manifesto ao avaliar ou inspecionar resultados. O `test.txt` distribuído com o BFDD corresponde a outra partição.

## Histórico e limitações

O primeiro experimento reservava cerca de 20% de cada combinação após embaralhamento. Fotografias próximas na sequência DJI podem retratar áreas sobrepostas ou próximas da mesma fachada, por isso foi adotado o corte contínuo por combinação no conjunto reservado. Uma alternativa com uma faixa intermediária embaralhada foi discutida e ficou sem implementação.

A partição garante separação por ID de imagem. A identidade de fachada, edifício ou sessão de captura ficou fora do agrupamento. Capturas próximas podem cruzar os subconjuntos, sobretudo entre combinações distintas. O registro histórico da inspeção visual identificou semelhanças em seis fronteiras. A distribuição de classes por imagem também difere da distribuição por pixel. As anotações originais foram mantidas, inclusive nos casos visualmente ambíguos entre descascamento, erosão e mancha.

O registro integral da preparação do BFDD permanece no histórico local, em `main:DATASET_ANALYSIS.md`.
