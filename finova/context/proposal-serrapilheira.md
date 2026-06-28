# Do evolutionary optimization algorithms shape the learning and behavior of artificial neural networks differently from traditional methods?

**Candidate:** Bruno Iochins Grisci

## Short Summary

I will use interpretability tools and controlled experiments on artificial neural networks to test whether neuroevolutionary and gradient-based optimization algorithms produce distinct learning patterns and internal representations. Although both may reach similar accuracy, they may lead to different types of errors, activation patterns, and weight configurations. Using standardized datasets, I will train networks with optimizers from both paradigms and compare them using relevance attribution techniques and visualization tools. The goal is to understand whether the optimizer influences how the network "thinks" and decides, helping to explain if the choice of optimization method shapes the behavior and reliability of neural networks in meaningful ways.

## Introduction

Artificial Neural Networks (ANNs) emerged as an innovative paradigm in artificial intelligence (AI), inspired in part by biological brains. The problem of training an ANN (optimizing its internal parameters) is inherently difficult. Given a generic ANN and a training set, finding a configuration of weights that correctly predicts all outputs is NP-hard. Nevertheless, a wide range of efficient ANN optimizers have been developed, relying on randomized, approximate, or local search algorithms.

For deep learning (ANNs with many layers), the standard approach is to use some variation of backpropagation and stochastic gradient descent (SGD). A less common yet long-standing approach is neuroevolution, developed in parallel with backpropagation since the 1980s. Neuroevolution refers to training ANNs using evolutionary computation (EC), a family of algorithms inspired by natural evolution and used to solve optimization problems. A key advantage of neuroevolution is that it does not require gradient information and can use arbitrary fitness functions. This flexibility has allowed it to thrive in domains such as reinforcement learning (RL), where it has achieved competitive results. In supervised learning, however, it has remained overshadowed by backpropagation, despite renewed interest in recent years. Despite their long history, gradient descent and neuroevolution have mostly evolved in isolation, only recently converging in areas like meta-learning. This creates a knowledge gap: few studies directly compare their effects on the final learned model, beyond performance metrics. Emerging results suggest they might shape networks in fundamentally different ways.

Ras et al. highlight the growing complexity of deep learning models (regularization, adaptivity, and architectural choices) as a major barrier to interpretability, reinforcing the need to better understand how training algorithms influence model behavior. Despite their widespread use across different fields, how gradient descent and EC influence ANN training remains under debate. Defining their relationship is an active area of research. Schmidt et al. compared several optimizers and found that performance varies by task.

SGD proponents often emphasize its dominance in deep learning, powering state-of-the-art models across diverse applications. In RL, ANNs are used to approximate value functions, embedded in a larger agent that learns through environmental interaction. Sutton and Barto argue that EC is only effective when the search space is small, or when enough time is available to explore it. Still, in some applications, neuroevolution has matched or even surpassed conventional methods. Besides deep learning, EC has also solved many hard problems more efficiently than gradient-based approaches, particularly via methods like differential evolution.

Some researchers have suggested that evolutionary strategies (ES) might just be noisy approximations of gradients based on finite differences. However, recent work shows that optimizing over a population (instead of a single solution) gives ES distinct properties. ES behaves like a noisy version of SGD, but still reaches high performance even at low correlation levels. The key difference is that ES optimizes the mean reward across the population, making it more robust to perturbations and able to explore regions of the search space that SGD might ignore. This makes ES particularly valuable in RL, where the true gradient is often inaccessible. In supervised learning, by contrast, gradients are known, reducing the appeal of EC. Martinez et al. provide a comprehensive review of neuroevolution's strengths and weaknesses for classification.

There is also a philosophical case for revisiting neuroevolution: since ANNs are inspired by biological brains, evolution, the mechanism that trained real brains, may offer complementary insights for training their artificial counterparts. However, neuroscience and cognitive psychology research still supports gradient-like mechanisms in some learning contexts.

This project hypothesizes that training the same ANN architecture with different optimizers leads to distinct activation patterns, error distributions, optimization trajectories, and even decision rules. If confirmed, this would represent a novel insight into how optimizers shape what and how ANNs learn. Even a negative result showing no substantial differences would be informative. It would suggest that neuroevolution, besides being less efficient, does not add interpretative value over standard gradient-based methods. Either way, the findings could help improve trust, robustness, and explainability in AI systems. Given the widespread adoption of ANNs in society, understanding how training methods affect their reliability and bias is a timely and important question.

## Hypothesis/Conjecture

My hypothesis is that training neural networks with neuroevolutionary algorithms produces distinct learning patterns and internal representations compared to traditional gradient-based methods, leading to differences in interpretability, generalization, and decision behavior. While gradient descent dominates deep learning due to its efficiency and scalability, it may converge to different solutions than those found by evolutionary strategies, which explore broader regions of the search space. Recent empirical studies suggest that the choice of optimizer can influence robustness and the kinds of features learned by the model. Previous works also found that in the context of reinforcement learning there may be behavioral differences between ANNs created with gradient or evolutionary training algorithms, and that the training algorithm impacts the final internal workings of the networks. Moreover, interpretability is needed to understand the internal "thought process" of large models with reliability, as their external behavior may be misleading. By comparing networks trained under identical conditions using both optimization approaches, this project seeks to uncover whether there are systematic and meaningful differences in learning patterns, potentially challenging the current view that the only metrics for optimizer choice are convergence speed and performance.

## Hypothesis/Conjecture Conception Risk

The main limitation is that the differences between gradient-based and neuroevolutionary optimizers may prove negligible or inconsistent across supervised learning tasks. While evolutionary strategies explore broader regions of the search space and optimize population-level objectives, recent studies suggest they may resemble noisy versions of gradient descent under certain conditions. It is possible that both methods converge to similar internal representations, especially in simple tasks or overparameterized models. Furthermore, the dataset or architecture might exert a stronger influence on model behavior than the optimizer itself, masking potential differences. In such cases, any observed divergence could be due to noise or hyperparameter sensitivity, rather than systematic effects of the optimization strategy.

## Approach

We will conduct controlled experiments comparing ANNs trained with gradient-based (e.g., SGD, Adam) and neuroevolutionary (e.g., LEEA, SHADE-ILS) optimization algorithms. LEEA is a neuroevolution strategy designed for the efficient training of large networks, while SHADE-ILS is a state-of-the-art differential evolution algorithm for continuous problems. We will collect data on performance, activation patterns, learned weights, and decision behavior. By training a statistically significant number of networks with each optimizer under identical conditions (architecture, initialization, loss function, training data, and target accuracy) we aim to isolate the effect of the optimizer. Interpretability tools such as Layer-wise Relevance Propagation, relevance aggregation, and dimensionality-reduction visualizations will be applied to analyze learned representations and prediction behavior. We will also evaluate the intersection and divergence of correct and incorrect classifications across optimizers. In early phases, we will focus on small image classification tasks using convolutional networks across standardized datasets, which allow easier training with neuroevolution and facilitate human interpretability. We will also include shortcut learning tests following the methodology of Geirhos et al., to assess whether the optimizers lead to different generalization failures. All experiments will be implemented using TensorFlow or PyTorch.

## Approach Risk

The approach may be inadequate if interpretability tools fail to detect meaningful differences in network behavior. Many popular attribution methods are sensitive to noise, model architecture, or hyperparameter settings, which may obscure the optimizer's effect. Even if differences exist, they may not be easily captured using post hoc visualizations. Furthermore, the controlled nature of the experiments, while necessary for isolating variables, may reduce generalization and limit the applicability of results to real-world settings. If both methods converge to similar predictions, differences may be subtle or task-specific. Alternative strategies include using simpler models to isolate effects more clearly, or applying mechanistic interpretability techniques that analyze learned functions at a more fundamental level, offering deeper insight into internal computation.

## Technical Risk

The main technical challenge is the high computational cost of neuroevolution, which is significantly slower and more resource-intensive than gradient-based methods. This may limit the number of training runs or the complexity of models used. To address this, we will rely on parallel computing, GPU-accelerated libraries like PyTorch, and efficient implementations of evolutionary algorithms. Another challenge is ensuring that all models, regardless of the optimizer used, reach comparable levels of predictive accuracy. This is essential to isolate behavioral differences unrelated to raw performance. Achieving this balance may require repeated hyperparameter tuning or adaptive stopping criteria. If resource constraints persist, we may reduce model size, simplify architectures, or use smaller datasets such as MNIST, for which training is faster and more easily interpretable.

## Originality

The originality lies in the hypothesis. To the best of my knowledge, this is the first systematic comparison of how gradient-based and evolutionary optimization algorithms influence the internal behavior and interpretability of neural networks. While most deep learning research focuses on performance, this project investigates whether different optimizers lead to qualitatively distinct representations even when predictive accuracy is similar. Moreover, gradient-based methods dominate deep learning and neuroevolution remains underexplored, especially in terms of its impact on interpretability.

## Impact

This project matters mainly to researchers in machine learning and optimization, but also to fields like health and finance that require robust, interpretable AI. This is reflected in the profile of the collaboration network described below. If successful, it will provide new insights into how optimizers shape neural network behavior and guide the development of hybrid algorithms. The free and open release of our code and data will support open science and foster collaboration. The project also contributes to Brazil's leadership in responsible and transparent artificial intelligence.

## Origin

This idea stemmed from my PhD on how ANNs select and weigh inputs from large datasets using interpretability techniques, and from my Master's research applying neuroevolution to identify key genes in gene expression data. I first encountered neuroevolution during my research stay at the University of Birmingham, where I trained models to mimic shark hunting behavior. These experiences sparked my curiosity about how different optimization strategies influence what ANNs learn and, along with my broader interest in natural evolution and recent debates on responsible AI, inspired this proposal.

## Team

The project will be conducted primarily by three faculty members and graduate and undergraduate students from the Federal University of Rio Grande do Sul (UFRGS). Two MSc students and one undergraduate student already work with Prof. Bruno Grisci on related topics and could be brought onboard, and new students will be recruited if the project is funded. The team has experience advising students from Computer Science, Computer Engineering, and Biotechnology.

Prof. Dr. Bruno Iochins Grisci (project lead) is a professor at the Department of Theoretical Informatics and advisor in the Graduate Program in Computing (PPGC) at UFRGS. He holds a PhD in Computer Science focused on machine learning interpretability and bioinformatics, and has published extensively on these topics and in neuroevolution. He will oversee all phases of the project, including design, implementation, analysis, and supervision.

Prof. Dr. Dennis Giovani Balreira is a faculty member in the Department of Applied Informatics and PPGC at UFRGS, with research focus in NLP, LLM, and machine learning. He will contribute to project design and student supervision.

Prof. Dr. Henrique Becker is also a professor at the Department of Theoretical Informatics of UFRGS and is an expert in combinatorial optimization, besides bringing academic and industry experience, including from Amazon. He will support implementation and infrastructure, including the setup of experimental environments.

## Current Collaboration Network

The project proponent already collaborates with Prof. Dr. Bruno César Feltes (UFRGS), who works on bioinformatics and optimization algorithms, and Prof. Dr. Rodrigo Ligabue Braun (UFCSPA), from the Department of Pharmaceutical Sciences, whose research also involves optimization methods and biological data analysis. While bioinformatics is not the main focus of this project, the findings and tools developed are expected to be applicable to biological datasets, extending the relevance and impact of the research. The Institute of Informatics of UFRGS also hosts many faculty members and graduate students with expertise in artificial intelligence and deep learning.

## Expansion of Collaboration Networks

The project network may expand through collaborations with researchers with whom the team has partnered in the past, including Prof. Hugo Verli (UFRGS) on modeling, and Prof. Gabriel de Oliveira Ramos (UNISINOS) on multi-objective optimization. Internationally, I may reactivate and strengthen ties with Profs. Inostroza-Ponta and Villalobos-Cid (USACH, Chile), Dr. Shan He (University of Birmingham, UK), Dr. Mathias Krause (KIT, Germany), and Prof. Evangelos Milios (Dalhousie, Canada), all experienced in optimization and machine learning. We also plan to recruit new MSc and undergraduate students, funded partly by institutional scholarships and partly through the project's personnel budget.

## Schedule

- **Year 1:** Conduct literature review on optimization and interpretability in neural networks. Define experimental protocols, select benchmark datasets, and implement baseline optimizers (for instance, Adam, SGD, LEEA, and SHADE-ILS). Hire and train students. Set up computing infrastructure. Launch internal documentation and project website.
- **Year 2:** Develop and implement interpretability tools (activation maximization, visualization tools, relevance aggregation, SHAP, etc.). Run initial experiments under controlled conditions. Validate training procedures and begin data collection. Submit a review or survey manuscript. Refine experimental design based on early results. Conduct internal presentations of preliminary findings.
- **Year 3:** Release first prototype of open-source analysis tools. Explore adjustments to neuroevolutionary algorithms for improved accuracy and time and memory efficiency. Draft and submit a research manuscript based on comparative results. Participate in academic events to gather feedback and foster collaboration.
- **Year 4:** Continue experimentation, including exploratory studies and robustness checks. Update analysis framework based on insights from previous stages. Submit additional manuscripts. Prepare tutorials or documentation for the developed tools. Organize dissemination through seminars, workshops, or community platforms. Extend experiments to new models or datasets.
- **Year 5:** Complete final experiments and consolidate findings. Submit final technical and financial reports. Release public-facing software, tools, and datasets. Deliver outreach materials (e.g., talk slides, blog posts, simplified visualizations). Host small seminars or webinars to present outcomes. Submit final manuscripts. Develop follow-up research plans for future work.

## Simplified Budget

- **Personnel:** R$ 184.800,00
- **Equipment and infrastructure:** R$ 40.200,00
- **Traveling expenses:** R$ 0,00
- **Experiments:** R$ 0,00
- **Publication fees:** R$ 0,00
- **Overhead:** R$ 25.000,00
- **Total:** R$ 250.000,00

## Bibliographic References

1. JUDD, J Stephen. Neural network design and the complexity of learning. MIT press; 1990.
2. STANLEY, Kenneth O. et al. Designing neural networks through neuroevolution. Nat. Mach. Intell., Springer, v. 1, n. 1, p. 24-35, 2019.
3. MORSE, Gregory; STANLEY, Kenneth O. Simple evolutionary optimization can rival stochastic gradient descent in neural networks. In: ACM. PROCEEDINGS of GECCO. 2016. P. 477?484.
4. ZHANG, Xingwen et al. On the Relationship Between the OpenAI Evolution Strategy and Stochastic Gradient Descent. arXiv preprint arXiv:1712.06564, 2017.
5. LEHMAN, Joel et al. ES is more than just a traditional finite-difference approximator. In: ACM. PROCEEDINGS of GECCO, 2018. P. 450-457.
6. SUCH, Felipe Petroski et al. An atari model zoo for analyzing, visualizing, and comparing deep reinforcement learning agents. arXiv preprint arXiv:1812.07069, 2018.
7. GEIRHOS, Robert et al. Shortcut learning in deep neural networks. Nat. Mach. Intell., Nature Publishing Group UK London, v. 2, n. 11, p. 665-673, 2020.
8. RAS, Gabrielle et al. Explanation methods in deep learning: Users, values, concerns and challenges. In: EXPLAINABLE and Interpretable Models in Computer Vision and Machine Learning. Springer, 2018. P. 19-36.
9. PASCANU, Razvan et al. On the saddle point problem for non-convex optimization. arXiv preprint arXiv:1405.4604, 2014.
10. SCHMIDT, Robin M.; SCHNEIDER, Frank; HENNIG, Philipp. Descending through a Crowded Valley - Benchmarking Deep Learning Optimizers, 2020. arXiv: 2007.01547 [cs.LG].
11. HU, Jie et al. Squeeze-and-excitation networks. In: IEEE CVPR, 2018. P. 7132?7141.
12. SILVER, D; SCHRITTWIESER, J; SIMONYAN, K et al. Mastering the game of go without human knowledge. Nature, Nature Publishing Group, v. 550, n. 7676, p. 354, 2017.
13. SUTTON, Richard S; BARTO, Andrew G. Reinforcement learning: An introduction. Cambridge, MA: MIT Press, 2018.
14. RISI, Sebastian; TOGELIUS, Julian. Neuroevolution in games: State of the art and open challenges. IEEE Transactions on Computational Intelligence and AI in Games, IEEE, v. 9, n. 1, p. 25-41, 2017.
15. YAMAN, Anil et al. Limited evaluation cooperative co-evolutionary differential evolution for large-scale neuroevolution. In: ACM. PROCEEDINGS of the Genetic and Evolutionary Computation Conference. 2018. P. 569-576.
16. WIERSTRA, Daan et al. Natural evolution strategies. J. Mach. Learn. Res., JMLR. org, v. 15, n. 1, p. 949980, 2014.
17. EBRAHIMI, Sayna; ROHRBACH, Anna; DARRELL, Trevor. Gradient-free policy architecture search and adaptation. arXiv preprint arXiv:1710.05958, 2017.
18. MARTINEZ, Aritz D et al. Lights and shadows in Evolutionary Deep Learning: Taxonomy, critical methodological analysis, cases of study, learned lessons, recommendations and challenges. Information Fusion, Elsevier, 2020.
19. LINDSEY, Jack et al. On the Biology of a Large Language Model. Transformer Circuits, 2025.
20. GRISCI, Bruno Iochins; KRAUSE, Mathias J; DORN, Marcio. Relevance aggregation for neural networks interpretability and knowledge discovery on tabular data. Inf. Sci., Elsevier, v. 559, p. 111-129, 2021.
21. RAI, Daking, et al. A practical review of mechanistic interpretability for transformer-based language models. arXiv preprint arXiv:2407.02646, 2024.
22. GRISCI, Bruno Iochins et al. Neuroevolution as a tool for microarray gene expression pattern identification in cancer research. J. Biomed. Inform., Elsevier, v. 89, p. 122-133, 2019.
23. NUNES, Rafael Oleques et al. A Named Entity Recognition Approach for Portuguese Legislative Texts Using Self-Learning. In: PROCEEDINGS of the 16th International Conference on Computational Processing of Portuguese, 2024. P. 290-300.
24. BECKER, Henrique et al. Comparative analysis of mathematical formulations for the two-dimensional guillotine cutting problem. International Transactions in Operational Research, v. 31, n. 5, p. 3010-3035, 2024.
