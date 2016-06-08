# Readings

## Introductory readings
- [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf): 12 Ideas generales acerca de Machine Learning
- [2015 Lisbon Machine Learning School](http://lxmls.it.pt/2015/?page_id=24): Slides de las charlas
- [P Value and the Theory of Hypothesis Testing: An Explanation for New Researchers](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2816758/pdf/11999_2009_Article_1164.pdf): Paper introductorio sobre la validación de hipotesis mediante p-values
- [Towards Machine Intelligence](http://arxiv.org/pdf/1603.08262v1.pdf): Ideas generales sobre los principios que debería seguir un algoritmo de aprendizaje de proposito general.

## Natural Language Understanding
- [Natural Language Understanding: Foundations and State-of-the-Art](http://icml.cc/2015/tutorials/icml2015-nlu-tutorial.pdf) by Percy Liang


## Neural Networks
#### General Ideas
- [A Survey: Time Travel in Deep Learning Space: An Introduction to Deep Learning Models and How Deep Learning Models Evolved from the Initial Ideas](http://arxiv.org/abs/1510.04781): Se trata de un survey que incluye el desarrollo de las redes neuronales a lo largo del tiempo, desde el perceptrón hasta los útlimos modelos basados en Deep RNN y DBN. Es muy interesante para comprender el "Big Picture" del area, conocer los diferentes modelos y sus aplicaciones
- [Evolution of Deep learning models](http://www.datasciencecentral.com/m/blogpost?id=6448529%3ABlogPost%3A305709): Recopilación de modelos de Redes Neuronales y su evolución a lo largo del tiempo.
- [Deep Learning (Libro)](http://www.deeplearningbook.org/): Se trata de un libro que está en proceso de desarrollo, se puede acceder a los capitulos ya escritos (está bastante avanzado). El libro está escrito por uno de los "gurus" del area (Yoshua Bengio) y se trata de una revisión general del area, entrando en detalle en cada uno de los modelos, su entrenamiento, optmización, etc...
- [Deep Learning Summer School, Montreal 2015](http://videolectures.net/deeplearning2015_montreal/): Escuela de Verano sobre Deep Learning
- [Hacker's guide to Neural Networks](http://karpathy.github.io/neuralnets/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)
- [Does Deep Learning Come from the Devil?](http://www.kdnuggets.com/2015/10/deep-learning-vapnik-einstein-devil-yandex-conference.html): Post "critico" con el tema, planteando alguno de los problemas del area y de su aplicación práctica por parte de la comunidad científica (e.g., Big Data and Deep Learning as Brute Force)
- [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://levyomer.files.wordpress.com/2015/03/improving-distributional-similarity-tacl-2015.pdf): Paper que presenta una idea interesante sobre Word Embeddings. Los autores afirman que lo que hacen es muy similar a otros métodos que ya existían en el estado del arte PMI y PPMI. Además prueban que el mejor rendimiento que tienen los primeros se debe únicamente a la parametrización de los algoritmos. De hecho, aplicando la misma parametrización a PPMI, se consiguen incluso mejores resultados que con Word Embeddings.
- [Entrevista a Yann Lecun](https://www.newscientist.com/article/dn28456-im-going-to-make-facebooks-ai-predict-what-happen-in-videos/): Entrevista con el Director de IA de Facebook
- [When Does Deep Learning Work Better Than SVMs or Random Forests?](http://www.kdnuggets.com/2016/04/deep-learning-vs-svm-random-forest.html): Breve post de KDnuggets al respecto
- [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](http://papers.nips.cc/paper/5486-identifying-and-attacking-the-saddle-point-problem-in-high-dimensional-non-convex-optimization.pdf): Paper del grupo de Yoshua Bengio que presenta una idea interesante acerca de los problemas de optimización (en el ámbito de las redes neuronales) en espacion multidimensionales que no son convexos, basado en el concepto de saddle point.

#### Specific applications
- [Natural Language Processing (Almost) from Scratch](http://arxiv.org/abs/1103.0398): El paper es un poco antiguo (2011), pero plantea una idea interesante. Trata de crear una suerte de modelo unificado para tareas de NLP (en concreto POS, Chunking, NER y Semantic Role Labelling) aplicando redes neuronales.
- [Text Understanding from Scratch](https://arxiv.org/pdf/1502.01710v5.pdf) NLU (Topic detection) utilizando redes neuronales a nivel de caracter.
- [Deep Learning: Going Beyond Machine Learning](https://www.youtube.com/watch?v=Ra6m70d3t0o): Charla de Amazon sobre Deep Learning y sus aplicaciones.
- [Deep Learning for Natural Language Processing](http://21ct.com/blog/deep-learning-for-natural-language-processing/): Blog post hablando del estado del arte de DL for NLP e ideas acerca de futuros desarrollos
- [Understanding Convolutional Neural Networks for NLP](http://www.kdnuggets.com/2015/11/understanding-convolutional-neural-networks-nlp.html): Blog post que habla sobre el uso cd CNN aplicadas a NLP. Explica brevemente la operación de convolución y la idea de CNN aplicada a Image Classification. A continuación explica la idea tras su aplicación para NLP. Finalmente recopila algunos ejemplos interesantes en el estado del arte.
- [Top 5 arXiv Deep Learning Papers, Explained](http://www.kdnuggets.com/2015/10/top-arxiv-deep-learning-papers-explained.html/2)
- [Deep Learning in Neural Networks: An Overview](http://arxiv.org/pdf/1404.7828v4.pdf)
- [CRF as RNN](http://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf): Paper que presenta una conexión entre Conditional Random Fields y Recurrent Neural Networks

##Reinforcement Learning
- [Reinforcement Learning: An Introduction](http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf): Libro de introducción al tema
- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html): Reinforced learning aplicado a Space Invaders
- [Demystifying Deep Reinforcement Learning](http://www.nervanasys.com/demystifying-deep-reinforcement-learning/): Introducción a Deep Reinforcement Learning
- [Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html): Ejemplo de aplicación de Deep Reinforcement Learning

## Probability
- [Probability, Paradox, and the Reasonable Person Principle](http://nbviewer.ipython.org/url/norvig.com/ipython/Probability.ipynb)


## word2vec
- [word2vec Tutorial](http://rare-technologies.com/word2vec-tutorial/)
- [Making Sense of word2vec](http://rare-technologies.com/making-sense-of-word2vec/)

## Generative vs Discriminative
- [On Discriminative vs. Generative classifiers](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)
- [Machine learning (Stanford)](https://www.youtube.com/watch?v=qRJ3GKMOFrE)

## Generative models
- [Structured Generative Models of Natural Source Code](http://jmlr.org/proceedings/papers/v32/maddison14.pdf)
- [A fast and simple algorithm for training neural probabilistic
language models](https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf)

## Other stuff
- [Visual Information Theory](http://colah.github.io/posts/2015-09-Visual-Information/)

## Other Reading Groups
- [Stanford](http://nlp.stanford.edu/read/)
- [Edinburgh](https://wiki.inf.ed.ac.uk/MLforNLP/WebHome)
- [Cambridge](http://www.wiki.cl.cam.ac.uk/rowiki/NaturalLanguage/ReadingGroup)
- [Heriot-Watt](https://sites.google.com/site/hwmlreadinggroup/)
- [Toronto](http://learning.cs.toronto.edu/mlreading.html)
