from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

popular_df = pickle.load(open('output/popular.pkl','rb'))
model_best = pickle.load(open('output/model_best.pkl','rb'))
book_copy = pickle.load(open('output/books.pkl','rb'))
rating_data = pickle.load(open('output/rating.pkl','rb'))
books_with_genres = pickle.load(open('output/books_with_genres.pkl','rb'))


app = Flask(__name__)


# get_unrated_book_ids, predict_and_sort_ratings, and get_top_predicted_books functions
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/top_book')
def top_ui():
    return render_template('top_book.html',
                            book_name = list(popular_df['original_title'].values),
                            authors = list(popular_df['authors'].values),
                            image = list(popular_df['image_url'].values),
                            votes = list(popular_df['rating_count'].values),
                            rating = list(popular_df['mean_rating'].values),
                            year = list(popular_df['original_publication_year'].values)
                          )

@app.route('/filter_book')
def filter_ui():
    genres = ["Art", "Biography", "Business", "Children", "Christian", "Classics", "Comics",  
          "Contemporary", "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction", "History", 
          "Horror", "Manga", "Memoir", "Music", "Mystery", "Nonfiction", "Paranormal", "Philosophy",
          "Poetry", "Psychology", "Religion", "Romance", "Science", "Science Fiction",   
          "Spirituality", "Sports", "Suspense", "Thriller", "Travel"]
    return render_template('filter.html', genres=genres)

@app.route('/filtered-books', methods=['POST'])
def filtered_books():
    genres = ["Art", "Biography", "Business", "Children", "Christian", "Classics", "Comics",  
          "Contemporary", "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction", "History", 
          "Horror", "Manga", "Memoir", "Music", "Mystery", "Nonfiction", "Paranormal", "Philosophy",
          "Poetry", "Psychology", "Religion", "Romance", "Science", "Science Fiction",   
          "Spirituality", "Sports", "Suspense", "Thriller", "Travel"]

    selected_genre = request.form.get('genre') 


    # filter book by genre
    if selected_genre == 'all':
        filtered_books = books_with_genres
    else:
        filtered_books = books_with_genres[books_with_genres['genres'].apply(lambda x: selected_genre.lower() in x)]

    return render_template('filtered_books.html', genres=genres, books=filtered_books, selected_genre=selected_genre)


@app.route('/recommend_book')
def recommend_ui():
    return render_template('recommendation.html')

@app.route('/recommendation', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form['user_id'])
        
        # Check if the user ID is non-positive (negative or zero)
        if user_id <= 0:
            return render_template('recommendation.html', user_id=None, recommended_books=None)

        # Get top predicted books for the user
        recommended_books = get_top_predicted_books(model_best, 9, user_id, rating_data, book_copy)

        return render_template('recommendation.html', user_id=user_id, recommended_books=recommended_books)
    except ValueError:
        return render_template('recommendation.html', user_id=None, recommended_books=None)
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)
