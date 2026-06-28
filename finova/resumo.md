# Resumo das atividades

- **Título do projeto:** 45624 - Análise e Comparação de Otimizadores de Redes Neurais Artificiais
- **Estudante:** Bruno Samuel Ardenghi Gonçalves
- **Orientador:** Prof. Dr. Bruno Iochins Grisci

No período da bolsa, foi desenvolvido um protótipo de ferramenta para executar, registrar, visualizar e comparar treinamentos de redes neurais artificiais otimizadas por métodos tradicionais e evolutivos. A atividade integrou o projeto de pesquisa sobre como a escolha do otimizador influencia não apenas a acurácia, mas também as trajetórias de aprendizado, os padrões internos e a interpretabilidade dos modelos.

O trabalho foi iniciado pelo estudo do problema, pela definição dos requisitos do protótipo e pela organização da arquitetura do sistema. Em seguida, foram implementados módulos de _backend_ em FastAPI e PyTorch para configurar experimentos com os _datasets_ MNIST e CIFAR-10, construir arquiteturas convolucionais compatíveis, executar treinamentos com SGD e LEEA, registrar métricas de perda e acurácia, controlar _seeds_ e dispositivos de execução, e salvar _checkpoints_ com metadados de reprodutibilidade. Também foram desenvolvidas rotinas de pausa, retomada e execução por linha de comando, incluindo suporte a sessões _tmux_ para treinamentos longos em ambiente remoto.

No _frontend_, foi construído um painel em React que permite configurar hiperparâmetros, acompanhar a telemetria em tempo real, carregar _checkpoints_ e comparar resultados. A etapa mais avançada do protótipo foi a criação de um módulo de análise comparativa entre dois _checkpoints_ compatíveis. Esse módulo gera relatórios com métricas de classificação, matrizes de confusão, calibração, sobreposição de acertos e erros, projeções PCA e t-SNE das representações internas, mapas de relevância por LRP, estatísticas de ativações e pesos, além de curvas de robustez a ruído, brilho e oclusão central.

As atividades incluíram ainda testes automatizados para os módulos de dados, modelos, otimizadores, API, _checkpoints_ e interface de linha de comando, bem como documentação básica de uso. Como resultado, o projeto avançou de uma formulação conceitual para um protótipo funcional e validável em laboratório, contribuindo para a construção de uma ferramenta aberta de apoio à análise de redes neurais e para a formação tecnológica em inteligência artificial interpretável.
