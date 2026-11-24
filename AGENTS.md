# Diretrizes de desenvolvimento

Estas diretrizes se aplicam a toda a implementação da branch `feat/classificacao-por-regioes`. Este é um projeto acadêmico. Priorize a compreensão do código, a clareza metodológica e a manutenção, preservando a correção matemática e a eficiência computacional.

## Organização e arquitetura

- Organize o código em módulos com responsabilidades bem definidas. Separe processamento de imagens, extração de características, treinamento, inferência e avaliação.
- Mantenha os scripts de execução separados dos módulos que implementam os algoritmos.
- Evite arquivos excessivamente extensos e funções com várias responsabilidades.
- Centralize os parâmetros e as configurações compartilhados. Reutilize funções existentes quando forem adequadas à tarefa.
- Use abstrações, classes e padrões de projeto quando simplificarem a implementação ou sua manutenção.

## Legibilidade

- Use identificadores descritivos em inglês. Evite abreviações pouco intuitivas.
- Escreva funções pequenas e objetivas. Prefira soluções simples e explícitas quando tiverem desempenho adequado.
- Evite expressões compactas que dificultem a leitura. Use anotações de tipo quando esclarecerem as entradas e saídas.
- Siga as convenções da PEP 8.

## Comentários e documentação

- Escreva comentários e docstrings em português. Documente o objetivo, os parâmetros, o retorno e os casos especiais das funções principais.
- Explique as etapas do algoritmo cuja finalidade exige contexto, especialmente operações matemáticas, manipulações de matrizes e operações vetorizadas.
- Justifique decisões técnicas relevantes e associe a implementação aos conceitos de Processamento de Imagens quando pertinente.
- Evite comentários que apenas repitam o código. Mantenha a documentação coerente com a implementação.
- Escreva para estudantes de Ciência da Computação familiarizados com Python, Processamento de Imagens e aprendizado de máquina. Explique os detalhes das bibliotecas e do algoritmo necessários para entender implementações mais complexas.

## Execução e resultados

- Identifique as etapas do pipeline nas mensagens do terminal. Apresente estatísticas e métricas de forma padronizada e compreensível.
- Use mensagens de erro que ajudem a identificar a causa do problema.
- Documente os comandos e parâmetros necessários para executar cada etapa e reproduzir os experimentos.

## Refatorações

Apresente previamente a necessidade, o escopo e o impacto de refatorações extensivas em módulos já implementados e testados. Considere o prazo do projeto e aguarde autorização antes de executá-las.
