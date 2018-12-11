# Recommendation Engine for Articles
This project was originally part of Udacity's Data Scientist 
nanodegree. The development notebook included in this repository 
contained the basic steps to create the methods that inform
the Recommender class I created. Although this project was originally
meant for IBM Watson's platform, I have generalized the approach
to any corpus of articles that ought to be served to users.

The goal of this module is to simplify the creation of a simple
recommender system. Anyone with two csv files containing
information about the interactions between users and existing articles
as well as a description about these articles is able to create
a recommendation engine with a few lines of code thanks to this project.


Certain article names weird because of formatting in original doc, normal
Specify developed on Mac.

## Installation

In order to install this package, simply type
`pip install article_recommender` in the terminal. Then, the main class can be imported
by `from article_recommender import ArticleRecommender`.
In order to run everything properly (including the development notebook), the
following libraries need to be installed locally:
* numpy
* pandas
* nltk
* sklearn
* re
* matplotlib
* pickle

## Structure

Two main folders are included in this repository.

### Development Notebook

This folder contains all the files pertaining to the development 
notebook that forms the basis of this package. This notebook was
created as part of Udacity's Data Scientist nanodegree, and all
the tests associated were created by them.

**project_tests.py**: tests developed by Udacity to test the functions created in the
development notebook.

**top_5.p**, **top_10.p**, **top_20.p**, **user_item_matrix.p**: additional documents
called in *project_tests.py* to test our results

**Recommendations_with_IBM.ipynb**: a Jupyter development notebook which contains all the
functions informing the package. The bare bones of this (instructions) were provided by
Udacity, but all the code is mine. Running this notebook as is will enable to follow
the main steps behind the recommendation engine, as well as test new functionalities

**Recommendations_with_IBM.html**: an HTML version of the development notebook, in case
only the results are of interest

#### Data

**articles_community.csv**: this example csv file contains 
information about the IBM Watson articles (we are particularly
interested in the *article_id* and *doc_description* fields).
This file is leveraged in the development notebook.

**user-item-interactions.csv**: this example csv file contains 
information about past interactions between users and articles
(we are particularly interested in the *user_id* and *article_id*
fields). This file isn't leveraged in the development notebook
per se but serves as an archetypal example in the code below.

**user-item-interactions.csv**: this example csv file is very similar
to the previous one, the only difference being that users are identified
by hashed emails rather than user ID's. This file is leveraged by
the development notebook.

## Usage
### Caveats

## Credit

## License