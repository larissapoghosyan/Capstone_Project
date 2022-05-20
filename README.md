# Optimality of state-of-the-art language representation models in NLP applications

The work is done during Larisa Poghosyan's capstone project for the degree of BS in Data Science in the Zaven and Sona Akian College of Science and Engineering.
Supervised by Vahe Hakobyan.

## Instructions

We provide the precomputed embeddings for each model on each dataset discussed in the project.
You can find the assets at: https://github.com/larissapoghosyan/Capstone_Project/releases/tag/embeddings

Otherwise, you can easily replicate the results presented in the work by following the instructions below.
See the corresponding sections for details.

### Notebooks

`Transformers_BERT_embeddings_both_datasets` notebook provides a pipeline to extract `BERT` embeddings and save them as h5py files.

The notebooks `W2V_FastText_embeddings_both_datasets` and `ELMo_embeddings_both_datasets` provide embeddings of `Word2Vec`, `FastText` and `ELMo` respectively.

The notebook `baseline_classifiers_IMDb_Dataset` contains the code to calculate point estimates for all the models in list on _IMDb Movie Reviews Sentiment Classification_ task.

`Pyro_for_pp_IMDb` contains the probabilistic programming pipeline using `pyro`. This notebook also provides the accuracy distributions and HDI plots, for IMDb dataset.

### Data

TODO IMDb data retrieval - https://github.com/larissapoghosyan/Capstone_Project/releases/download/embeddings/IMDb_Reviews.csv

TODO Word2Vec repo and cmd - api.load('word2vec-google-news-300')

TODO FastText repo and cmd -- !wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
                              !unzip wiki-news-300d-1M.vec.zip

TODO ELMo repo and cmd -- !pip install allennlp==0.9.0
                          !pip install flair
                          !pip install sacremoses

TODO TinyBERT, BERT, RoBERTa repo and cmd -- !pip install transformers (HuggingFace pre-trained models)

### Requirements

All of the experiments are performed on `Colab` and `Colab Pro+` on a single (K40 / V100 / P100) GPU.
Using GPUs for the project, however, is not required but is expected to take much longer.
For example, experiments on `ELMO` takes approximately `1 hour` on `K40` GPU while on CPU it takes up to `13 hours`.

TODO system requirements and python requirements

### Hardware

TODO Hardware used for training. About Colab and Colab Pro+. GPU requirements.

## Abstract

Pre-trained models like BERT, Word2Vec, FastText, etc., are widely used in many NLP applications such as chatbots, text classification, machine translation, etc. Such models are trained on huge corpora of text data and can capture statistical, semantic, and relational properties of the language. As a result, they provide numeric representations of text tokens (words, sentences) that can be used in downstream tasks. Having such pre-trained models off the shelf is convenient in practice, as it may not be possible to obtain good quality representations by training them from scratch due to lack of data or resource constraints.
That said, in the practical setting, such embeddings are often used as inputs to models to serve the purpose of the task. For example, in a sentence classification task, it is possible to use a Logistic Regression on the top average Word2Vec embeddings. Using such embeddings on real-life industrial problems could produce some optimistic improvements over baselines; however, it is not clear whether those improvements are reliable or not. 
In our study, we intend to check the question at hand by formulating multiple applicable and viable tasks in the industry and replicating the workflow of data scientists. Our goal is to construct various models (different in sophistication) that use embeddings as inputs and use a methodology to report the confidence bounds of the metric of interest.
With this experiment we hope to develop an understanding of the phenomenon of having optimal results on the paper that might not be optimal in reality; thus, aiming to find a reliable method, that will aid the decision making process and facilitate the model selection.

<img width="551" alt="kde" src="https://user-images.githubusercontent.com/43134338/169176706-1462de85-2fa8-4fa7-b785-8cd7fe290384.png">
<sup>Kernel Density Estimation plots of Accuracy Distributions for each Model on IMDb Movie Sentiment Classification task</sup>


