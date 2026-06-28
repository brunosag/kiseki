# Roteiro para vídeo documentário

- **Tema:** Otimizadores de redes neurais artificiais
- **Aluno:** Bruno Ardenghi
- **Orientador:** Prof. Dr. Bruno Iochins Grisci
- **Duração-alvo:** 2min40s a 2min50s
- **Formato:** MP4, horizontal, 1280 x 720, até 20 MB.

## Estrutura

| Tempo | Imagem sugerida | Narração |
| --- | --- | --- |
| 0:00-0:12 | Tela dividida: a mesma imagem de um dígito ou objeto entra em duas redes neurais idênticas. Os caminhos se separam em dois métodos de treinamento. | "No treinamento de redes neurais, métodos de otimização diferentes podem produzir modelos com desempenhos semelhantes." |
| 0:12-0:28 | Close em gráficos de treinamento e, depois, uma caixa-preta se abrindo para revelar pesos, ativações e mapas de calor. | "Mesmo assim, esses modelos podem seguir trajetórias de aprendizagem distintas e organizar suas decisões de maneiras diferentes, o que exige análise além da acurácia." |
| 0:28-0:45 | Imagem em laboratório ou diante do dashboard. Inserir tema do trabalho e identificação do estudante. | "Para apoiar esse tipo de investigação, foi desenvolvido um protótipo para treinar, registrar e comparar redes neurais otimizadas por métodos baseados em gradiente e por neuroevolução." |
| 0:45-1:10 | Capturas do backend, terminal e painel web. Mostrar configuração de dados, método de treinamento, semente, tamanho de lote e dispositivo. | "A ferramenta possibilita configurar experimentos com bases de imagens padronizadas, executar treinamentos por métodos tradicionais e evolutivos, acompanhar métricas em tempo real e salvar registros com informações de reprodutibilidade." |
| 1:10-1:28 | Mostrar controle de pausa, retomada, lista de checkpoints e execução por terminal/tmux. | "Também foram implementados recursos para experimentos de longa duração, como pausa, retomada, checkpoints, execução por linha de comando e suporte a sessões remotas." |
| 1:28-1:55 | No dashboard, selecionar dois checkpoints e abrir o relatório de comparação. Mostrar matriz de confusão, calibração e sobreposição de acertos e erros. | "O módulo principal compara dois modelos treinados sob condições compatíveis. O relatório apresenta desempenho, padrões de erro, divergências de classificação e níveis de confiança das previsões." |
| 1:55-2:18 | Mostrar projeções das representações internas, mapas de relevância sobre imagens, estatísticas de ativações/pesos e curvas de robustez. | "Para ampliar a análise, são geradas projeções das representações internas, mapas visuais de relevância, estatísticas de ativações e pesos, e testes de robustez com ruído, alteração de brilho e oclusão." |
| 2:18-2:36 | Animação simples da metodologia: mesma arquitetura, mesmo conjunto de dados, otimizadores diferentes, análise final. | "A metodologia busca isolar o efeito do otimizador: são mantidos arquitetura e dados comparáveis, são treinados modelos por estratégias diferentes e as decisões são avaliadas no mesmo conjunto de teste." |
| 2:36-2:50 | Encerramento com painel funcionando e frase final na tela: "Métricas, comportamento e interpretabilidade". | "Como resultado, foi obtido um protótipo funcional para apoiar pesquisas em inteligência artificial interpretável e contribuir para análises mais transparentes do comportamento de redes neurais." |

## Observações de produção

- Usar linguagem direta, sem excesso de termos técnicos na narração.
- Priorizar imagens reais do protótipo: dashboard, gráficos, checkpoints, relatórios e mapas de relevância.
- Inserir poucos textos na tela: título, nomes, otimizadores comparados e uma frase-síntese por bloco.
- Gravar a fala em ritmo natural; o roteiro tem margem para cortes e pausas sem ultrapassar 3 minutos.
