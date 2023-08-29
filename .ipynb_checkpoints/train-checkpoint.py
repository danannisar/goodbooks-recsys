# For retrain model in full data

import pickle
import numpy as np
import pandas as pd

# Functions -------------------------------------------------------------

def load_data(book_path, rating_path):
    
    #reads the CSV file data and saves it as a DataFrame
    rating_data = pd.read_csv(rating_path, delimiter=',')
    book_data = pd.read_csv(book_path, delimiter=',')
    
    #copy dataframe book_data, and delete some feature.
    book_copy = book_data.copy()
    book_copy = book_copy.drop(columns=['best_book_id','work_id','books_count','isbn',
           'isbn13','title','language_code','average_rating',
           'ratings_count', 'work_ratings_count', 'work_text_reviews_count',
           'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
           'small_image_url'], axis=1)
    book_copy.head(3)
    
    #fill null values in book_data
    print("Missing values before fillna: ", book_copy.isnull().sum())
    book_copy['original_publication_year'] = book_copy['original_publication_year'].fillna(0)
    book_copy['original_title'] = book_copy['original_title'].fillna(book_data['title'])
    print("Missing values after fillna: ", book_copy.isnull().sum())
    
    #changes the data type original_publication_year column to int data type
    book_copy.loc[:, 'original_publication_year'] = book_copy['original_publication_year'].astype(int)
    book_copy.dtypes
    
    ## b. drop duplicated rows
    print("Books shape before drop dup: ", book_copy.shape)
    print("Ratings shape before drop dup: ", rating_data.shape)
    book_copy.drop_duplicates(subset = ['book_id', 'goodreads_book_id'], inplace = True)
    rating_data.drop_duplicates(subset=['user_id','book_id'], inplace = True)
    print("Books shape after drop dup: ", book_copy.shape)
    print("Ratings shape after drop dup: ", rating_data.shape)
    
    return book_copy, rating_data

def get_unrated_book_ids(rating_data, user_id):
    """
    Gets a list of book IDs that a user has not rated yet.

    Parameters
    ----------
    rating_data : DataFrame
        The DataFrame containing the rating data.
    user_id : int
        The ID of the user for whom we want to find unrated book IDs.

    Returns
    -------
    unrated_book_ids : set
        A set of book IDs that the user has not rated.
    """
    #get unique book_id
    unique_book_ids = set(rating_data['book_id'])
    #get book_id that is rated by user_id = 2
    rated_book_ids = set(rating_data.loc[rating_data['user_id'] == user_id, 'book_id'])
    #find unrated book_id
    unrated_book_ids = unique_book_ids.difference(rated_book_ids)
    
    return unrated_book_ids

def predict_and_sort_ratings(model, user_id, unrated_book_ids):
    """
    Predicts and sorts unrated books based on predicted ratings for a given user.

    Parameters
    ----------
    model : object
        The collaborative filtering model used for predictions.
    user_id : int
        The ID of the user for whom we want to predict and sort unrated books.
    unrated_book_ids : list
        A list of book IDs that the user has not rated yet.

    Returns
    -------
    predicted_unrated_book_df : DataFrame
        A DataFrame containing the predicted ratings and book IDs,
        sorted in descending order of predicted ratings.
    """

    #initialize
    predicted_unrated_book = {
        'user_id': user_id,
        'book_id': [],
        'predicted_rating': []
    }
    
    #loop all unrated book
    for book_id in unrated_book_ids:
        #make predict
        pred_id = model.predict(uid=predicted_unrated_book['user_id'],
                                iid=book_id)
        #append
        predicted_unrated_book['book_id'].append(book_id)
        predicted_unrated_book['predicted_rating'].append(pred_id.est)

    #create df
    predicted_unrated_book_df = pd.DataFrame(predicted_unrated_book).sort_values('predicted_rating',
                                                                                  ascending=False)

    return predicted_unrated_book_df

def get_top_predicted_books(model, k, user_id, rating_data, book_data):
    """
    Gets the top predicted books for a given user based on a collaborative filtering model.

    Parameters
    ----------
    model : object
        The collaborative filtering model used for predictions
    k : int
        The number of top predicted books to retrieve
    user_id : int
        The ID of the user for whom to get top predicted books
    rating_data : DataFrame
        The DataFrame containing the rating data
    book_data : DataFrame
        The DataFrame containing the book details

    Returns
    -------
    top_books_df : DataFrame
        A DataFrame containing the top predicted books along with their details
    """

    # Get unrated book IDs for the user
    unrated_book_ids = get_unrated_book_ids(rating_data, user_id)

    # Predict and sort unrated books
    predicted_books_df = predict_and_sort_ratings(model, user_id, unrated_book_ids)

    # Get the top k predicted books
    top_predicted_books = predicted_books_df.head(k).copy()

    # Add book details to the top predicted books
    top_predicted_books['authors'] = book_data.loc[top_predicted_books['book_id'], 'authors'].values
    top_predicted_books['original_publication_year'] = book_data.loc[top_predicted_books['book_id'], 'original_publication_year'].values
    top_predicted_books['original_title'] = book_data.loc[top_predicted_books['book_id'], 'original_title'].values
    top_predicted_books['image_url'] = book_data.loc[top_predicted_books['book_id'], 'image_url'].values

    return top_predicted_books

# ----------------------------------------------------------------------------
#load data from path
rating_path = 'data/ratings.csv'
book_path = 'data/books.csv'

book_copy, rating_data = load_data(book_path, rating_path)
pickle.dump(book_data_small,open('books.pkl','wb'))
model_best = pickle.load(open('output/model_best.pkl', 'rb'))

