# Recommendation Engine for Articles
This project was originally part of Udacity's Data Scientist 
nanodegree. The development notebook included in this repository
contained the basic instructions to create the methods that inform
the ArticleRecommender class I created. Although this project was originally
designed to create a recommendation engine for articles on the
IBM Watson's platform, my solution works for any recommendation problem pertaining to
articles.

The goal of this module is to simplify the creation of a simple
recommender system. Anyone with two csv files containing
information about the interactions between users and existing articles
as well as a description about these articles is able to create
a recommendation engine with a few lines of code thanks to this project.

## Installation

In order to install this package, simply type
`pip install article_recommender` in the terminal. Then, the main class can be imported
by `from article_recommender import ArticleRecommender`. Below is some example code
including installation and import of the module.
In order to run everything properly (including the development notebook), the
following libraries need to be installed locally:
* numpy
* pandas
* nltk
* sklearn
* re
* matplotlib
* pickle
* scipy

## Structure

Two main folders are included in this repository.

### Development Notebook

This folder contains all the files pertaining to the development 
notebook, which was
created as part of Udacity's Data Scientist nanodegree, and all
the tests associated were created by them.

**project_tests.py**: tests developed by Udacity to test the functions created in the
development notebook

**top_5.p**, **top_10.p**, **top_20.p**, **user_item_matrix.p**: additional files
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

### ArticleRecommender

This folder contains all the files necessary to create and upload the package to PyPi.

**setup.py**, **dist**, **article_recommender.egg-info**: these files are not of interest
to understand the package, they only contain metadata or elements necessary for the upload
to PyPi.

#### Article Recommender

The folder article_recommender is where the main files are included.

**license.txt**: a license in order to use this software, using a template provided by
MIT

**setup.cfg**: additional metadata information

**recommender_helper_functions**: helper functions created from the development notebook

**recommender**: the main module creating the ArticleRecommender class 

## Usage

The ArticleRecommender class has 4 main methods.

### 1. load_data 

To load both interaction and content information, either from two csv files
or two existing Pandas dataframes. It also handles type conversion to make sure
all the following methods can be used.

        INPUT
        interactions_path - (str) path to a CSV file containing information about interactions
        between users and articles. It must include user_id (int), article_id (a float of the form
        20.0) and title
        
        content_path - (str) path to a CSV file containing information about the content of the
        articles. It must include article_id (as an int) and doc_description
        between users and articles. It must include user_id, article_id and title
        
        csv -(bool) a Boolean specifying whether the document is a CSV file or an already
        existing Pandas dataframe
        
        interactions - (pandas dataframe) contains information about interactions
        between users and articles. It must contain user_id, article_id (a string of the form
        '20.0') and title
        
        content -(pandas dataframe) contains information about the content of the
        articles. It must include article_id (as an int) and doc_description
        
        OUTPUT
        None, updates the following attributes
        self.df - (pandas dataframe) (pandas dataframe) contains information about interactions
        between users and articles
        self.df_content - (pandas dataframe) contains information about the content of the
        articles

### 2. fit 

This method performs matrix factorization using a basic form of FunkSVD with no regularization

        INPUT:
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations

        OUTPUT:
        None - stores the following as attributes:
        n_users - the number of users (int)
        n_articles - the number of articles (int)
        num_interactions - the number of interactions calculated (int)
        user_item_df - (pandas df) a user by item dataframe with interactions and nans for values
        user_item - (np array) a user by item numpy array with interactions and nans for values
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations
        user_ids_series - (series) all the user_id's contained in our dataset
        article_ids_series - (series) all the article id's contained in our dataset
        user_mat - (np array) the user matrix resulting from FunkSVD
        article_mat - (np array) the item matrix resulting from FunkSVD
        ranked_articles - (pandas dataframe) a dataframe with articles ranked by their number of interactions
        
### 3. predict_interactions 

To predict the number of interactions between a user ID and an article ID

        INPUT:
        user_id - (int) the user_id from interactions df
        article_id - (int) the article_id according the interactions df
        doc_description
        df_content - updated content dataframe with the new article to predict

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD

        Description: we have four cases that we want to treat differently:
            - if both the user_id and article_id are in the interactions df, we use the
            results from FunkSVD
            - if the user_id is in interactions df but the article ID is not (it is brand new),
            if it is in the content dataframe then we use content-based filtering to predict ratings
            based on ratings of similar movies
            - if the user_id is not in the interactions df, we cannot make any predictions
            - if the article_id is neither in the content or interaction dataset then we cannot predict anything
            
            
### 4. make_recommendations 

Given either a user or an article ID, make a number of recommendations

        INPUT:
        _id - either a user or movie id (int)
        _id_type - "article" or "user" (str)
        rec_num - number of recommendations to return (int)

        OUTPUT:
        recs - (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given

        DESCRIPTION:
        If the user is available in the interactions dataset, we use the matrix factorization data.
        If the user is new we simply return the top rec_num articles
        If we are trying to recommend based on an article, we will use a content-based recommendation system

### Example Code

Below is an example use of the class. The file structure used to load the data is that
which results from cloning the directory locally. Setting the current working directory
to "/recommendation_engine_ibm', and after installing the package:

```buildoutcfg
from article_recommender import ArticleRecommender

# Instantiate our recommender
rec = ArticleRecommender()

# Load the data, assuming it is in csv files
# To use already existing Pandas dataframe (after pre-processing e.g.), 
# see the method docstring
rec.load_data('development_notebook/data/user-item-interactions.csv', 
'development_notebook/data/articles_community.csv')

# Use the fit method to create user and article matrices, and find our top articles
rec.fit()

# Predict the number of interactions between an existing user and article
interactions_1_1160 = rec.predict_interactions(1, 1160)

# Predict the number of interactions between an existing user and a new article 
# This article needs to be in the content dataset, i.e. have a description we can use
# to make content recommendations. Here, this assumes an article with ID 123456 has
# been added to the content dataset
interactions_1_123456 = rec.predict_interactions(1, 123456)

# Make recommendations for a given user ID
recs_user_1 = rec.make_recommendations(_id=1)

# Make recommendations for a given article ID
recs_article_2 = rec.make_recommendations(_id=2, _id_type='article')
```

For more information on how each method works and all the options available, please refer
to the docstrings included above as well as in the appropriate files.

## Credit

The development notebook contained prompts and tests written by Udacity and were not my 
own. All the remainder of this repository is my own work and should only be used under
the appropriate license.