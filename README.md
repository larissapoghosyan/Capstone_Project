# Optimality of state-of-the-art language representation models in NLP applications

The work is done during Larisa Poghosyan's capstone project for the degree of BS in Data Science in the Zaven and Sona Akian College of Science and Engineering.

## Abstract

Pre-trained models like BERT, Word2Vec, FastText, etc., are widely used in many NLP applications such as chatbots, text classification, machine translation, etc. Such models are trained on huge corpora of text data and can capture statistical, semantic, and relational properties of the language. As a result, they provide numeric representations of text tokens (words, sentences) that can be used in downstream tasks. Having such pre-trained models off the shelf is convenient in practice, as it may not be possible to obtain good quality representations by training them from scratch due to lack of data or resource constraints.

That said, in the practical setting, such embeddings are often used as inputs to models to serve the purpose of the task. For example, in a sentence classification task, it is possible to use a Logistic Regression on the top average Word2Vec embeddings. Using such embeddings on real-life industrial problems could produce some optimistic improvements over baselines; however, it is not clear whether those improvements are reliable or not. It is yet to decide whether the problem lies within the data or the embedding method itself since the embedding could better adapt to noisy data rather than adequately learning regularities in the data and have a better generalization capability.

In our study, we intend to check the question at hand by formulating multiple applicable and viable tasks in the industry and replicating the workflow of data scientists. Our goal is to construct various models (different in sophistication) that use embeddings as inputs and use a methodology to report the confidence bounds of the metric of interest. Based on our learnings, we would like to explore possibilities of exposing errors in the data. Our experiment will carry quantitative and qualitative nature, as it may be hard to find a reliable measure to evaluate raw textual data. We hope to develop an understanding of the phenomenon of having optimal results on the paper that might not be optimal in reality; thus, aiming to find solutions to quickly identify the issue that might be hidden in the embedding itself as well as the quality or size of data.

<img width="444" alt="KDEplots_IMDb" src="https://user-images.githubusercontent.com/43134338/169170588-bedaf15a-e3f2-456f-8f53-4efd29d9f787.png">
<sup>Kernel Density Estimation plots of Accuracy Distributions fro each Model on IMDb Movie Sentiment Classification task</sup>

Other text
