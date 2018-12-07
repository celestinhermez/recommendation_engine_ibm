import numpy as np
import pandas as pd
import recommender_helper_functions as rf
import sys # can use sys to take command line arguments


# first create a prediction method
# if existing user-article pair, use FunkSVD results
# if new article, find similarity of this article to all others, find top 5 most similar
# and take average of their ratings. Make sure we have all the necessary information to find
# the similarity
# if brand new user, cannot predict a rating

# to make recommendations:
# if new user: knowledge-based
# if existing user: SVD top k rating and most popular overall to break ties, not risk recommending
# brand new articles for it would be very computationally expensive and too "risky"
# if article: content-based

# to predict rating:
# if new user: nothing
# if existing user and existing article: FUnkSVD
# if existing article and new article (specify brand new article), predict average
# of top similar ratings. Make sure we have all the information necessary to predict (doc desc etc)

# in the README, give examples of usage. Recommend pip installing, but possible to clone the repo

class Recommender():
    '''
    This Recommender uses a mix of FunkSVD and content-based methods to make recommendations
    for existing users, both for existing and brand new articles.
    For brand new users, in absence of any information the top rated articles (calculated as
    the number of interactions) will be recommended
    This recommender can also predict ratings for all user-article pairs (including brand new
    articles), with the exception of brand new users
    '''
    def __init__(self):
        '''
        We initialize all the attributes
        '''
        self.df = None
        self.df_content = None
        self.user_item_df = None
        self.user_item = None
        self.iters = None
        self.latent_features = None
        self.user_mat = None
        self.article_mat = None
        self.n_articles = None
        self.n_users = None
        self.learning_rate = None
        self.user_ids_series = None
        self.article_ids_series = None
        self.num_interactions = None
        self.ranked_articles = None

    def load_data(self, interactions_path, content_path, csv=True,
                  interactions=None, content=None):
        '''
        :param interactions_path: (str) path to a CSV file containing information about interactions
        between users and articles. It must include user_id, article_id (a float of the form
        20.0) and title
        :param content_path: (str) path to a CSV file containing information about the content of the
        articles. It must include article_id (as an int) and doc_description
        between users and articles. It must include user_id, article_id and title
        :param csv: (bool) a Boolean specifying whether the document is a CSV file or an already
        existing Pandas dataframe
        :param interactions: (pandas dataframe) contains information about interactions
        between users and articles. It must contain user_id, article_id (a string of the form
        '20.0') and title
        :param content: (pandas dataframe) contains information about the content of the
        articles. It must include article_id (as an int) and doc_description

        :return: None, updates the following attributes
        self.df - (pandas dataframe) (pandas dataframe) contains information about interactions
        between users and articles
        self.df_content - (pandas dataframe) contains information about the content of the
        articles
        '''

        if csv:
            self.df = pd.read_csv(interactions_path)
            self.df_content = pd.read_csv(content_path)

        else:
            self.df = interactions
            self.df_content = content

        if not self.df.article_id.dtype == 'float64':
            print('The article ID in the interactions dataset needs to be of the form "20.0". \
            Please modify and reload the data')

        if self.df_content.article_id.dtype == int:
            print('The article ID in the content dataset needs to be an integer. \
                        Please modify and reload the data')

    def fit(self, latent_features=12, learning_rate=0.0001, iters=100):
        '''
        This function performs matrix factorization using a basic form of FunkSVD with no regularization

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

        '''

        # Create user-item matrix
        self.user_item_df = rf.create_user_item_matrix(self.df)
        self.user_item = np.array(self.user_item_df)

        # Store more inputs
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Set up useful values to be used through the rest of the function
        self.n_users = self.user_item_df.shape[0]
        self.n_articles = self.user_item_df.shape[1]
        self.num_interactions = self.user_item_df.sum().sum()
        self.user_ids_series = np.array(self.user_item_df.index)
        self.article_ids_series = np.array(self.user_item_df.columns)

        # initialize the user and movie matrices with random values
        user_mat = np.random.rand(self.n_users, self.latent_features)
        article_mat = np.random.rand(self.latent_features, self.n_articles)

        # initialize sse at 0 for first iteration
        sse_accum = 0

        # keep track of iteration and MSE
        print("Optimization Statistics")
        print("Iterations | Mean Squared Error ")

        # for each iteration
        for iteration in range(self.iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # For each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_articles):

                    # if the rating exists
                    if self.user_item[i, j] > 0:

                        # compute the error as the actual minus the dot product of the user and movie latent features
                        diff = self.user_item[i, j] - np.dot(user_mat[i, :], article_mat[:, j])

                        # Keep track of the sum of squared errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(self.latent_features):
                            user_mat[i, k] += self.learning_rate * (2*diff*article_mat[k, j])
                            article_mat[k, j] += self.learning_rate * (2*diff*user_mat[i, k])

            # print results
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_interactions))

        # SVD based fit
        # Keep user_mat and movie_mat for safe keeping
        self.user_mat = user_mat
        self.article_mat = article_mat

        # Knowledge based fit
        self.ranked_articles = rf.rank_articles(self.df)

    def predict_interactions(self, user_id, article_id):
        '''
        INPUT:
        user_id - (int) the user_id from interactions df
        article_id - (int or float) the article_id according the interactions df
        doc_description
        df_content - updated content dataframe with the new article to predict

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD

        Description: we have two cases that we want to treat differently:
            - if both the user_id and article_id are in the interactions df, we use the
            results from FunkSVD
            - if the user_id is in interactions df but the article ID is not (it is brand new),
            if it is in the content dataframe then we use content-based filtering to predict ratings
            based on ratings of similar movies
            - if the user_id is not in the interactions df, we cannot make any recommendations
        '''

        # we first deal with the case where both user and article are in the interactions dataframe
        try:
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            article_col = np.where(self.article_ids_series == float(article_id))[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_mat[user_row, :], self.article_mat[:, article_col])

            article_name = rf.get_article_names([article_id], self.df)
            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(pred, 2), article_name))

            return pred

        except:
            None

        # except:
        #     # then we deal with the case when the user_id is in the interactions df but not
        #     # the article. In this case, we use content-based filtering to find the rating
        #     # based on rating of similar movies
        #     try:
        #         if article_id in self.df_content.article_id:
        #             similar_articles = rf.make_content_recs(article_id, self.df_content)
        #             ratings = self.ranked_articles.loc[self.ranked_articles.article_id.isin(
        #                 similar_articles
        #             ),'num_interactions']
        #             article_name = rf.get_article_names(article_id, self.df)
        #             pred = ratings.mean()
        #             print("For user {} we predict a {} rating for the movie {}.".format(user_id,
        #                                                                                 round(pred,2),
        #                                                                                 article_name))
        #
        #             return pred
        #
        #         else:
        #             print('The provided article ID is not in the interactions nor content dataset, \
        #                   no prediction can be made')
        #             return None
        #
        #     # for the final case, when the user ID is not in the interactions df we can't make predictions
        #     except:
        #         print('The user ID provided is not in the interactions df, no prediction can be made')
        #         return None


    def make_recommendations(self, _id, _id_type='movie', rec_num=5):
        '''
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        rec_num - number of recommendations to return (int)

        OUTPUT:
        recs - (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given
        '''
        # if the user is available from the matrix factorization data,
        # I will use this and rank movies based on the predicted values
        # For use with user indexing
        rec_ids, rec_names = None, None
        if _id_type == 'user':
            if _id in self.user_ids_series:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(self.user_ids_series == _id)[0][0]

                # take the dot product of that row and the V matrix
                preds = np.dot(self.user_mat[idx,:],self.movie_mat)

                # pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = self.movie_ids_series[indices]
                rec_names = rf.get_movie_names(rec_ids, self.movies)

            else:
                # if we don't have this user, give just top ratings back
                rec_names = rf.popular_recommendations(_id, rec_num, self.ranked_movies)
                print("Because this user wasn't in our database, we are giving back the top movie recommendations for all users.")

        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = list(rf.find_similar_movies(_id, self.movies))[:rec_num]
            else:
                print("That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")

        return rec_ids, rec_names

if __name__ == '__main__':
    import recommender as r

    #instantiate recommender
    rec = r.Recommender()

    # fit recommender
    rec.fit(reviews_pth='data/train_data.csv', movies_pth= 'data/movies_clean.csv', learning_rate=.01, iters=1)

    # predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    print(rec.make_recommendations(8,'user')) # user in the dataset
    print(rec.make_recommendations(1,'user')) # user not in dataset
    print(rec.make_recommendations(1853728)) # movie in the dataset
    print(rec.make_recommendations(1)) # movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.num_ratings)
