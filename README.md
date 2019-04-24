# deep-learning-nlp
Resources and notebook for ["Deep Learning and Modern NLP" tutorial](https://databricks.com/sparkaisummit/north-america/sessions-single-2019?id=168) for Spark + AI Summit 2019

## Environment Setup

This tutorial requires an existing installation of [Anaconda 3](https://www.anaconda.com/download/#macos) (tested with Python 3.6). From the root directory of the repo, run:

```
conda env create -f environment.yml
source activate deep-learning-nlp
```

## Datasets used in Tutorials
Data for these tutorials are sourced from various locations, and prepared in advance into pickled Pandas DataFrames. The original sources of the data can be found in the links below. 

* Perceptron
	* [Stanford Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
	* [Stack Overflow Q&A](https://cloud.google.com/blog/products/gcp/google-bigquery-public-datasets-now-include-stack-overflow-q-a)
* LSTM Classification
	* [R8 of Reuters 21578](https://www.cs.umb.edu/~smimarog/textmining/datasets/)
* Part of Speech Tagging
	* Sample of the [Penn Treebank](https://corochann.com/penn-tree-bank-ptb-dataset-introduction-1456.html) dataset from the [NLTK Corpora](http://www.nltk.org/nltk_data/)
* Machine Translation
	* [European Parliament Proceedings Parallel Corpus 1996-2011](http://www.statmt.org/europarl/)
