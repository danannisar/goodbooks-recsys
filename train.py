# For retrain model in full data

import pickle
import numpy as np
import pandas as pd

# Load  Dataset -------------------------------------------------------------

#load data from path
rating_path = 'data/ratings.csv'
book_path = 'data/books.csv'
book_tags_path = 'data/book_tags.csv'
tags_path = 'data/tags.csv'

def load_data(book_path, rating_path):
    """
    
    Function to load book and rating data
    - subsetting only the used columns
    - fill in missing values
    - drop duplicate rows
    
    """
        
    # reads the CSV file data and saves it as a DataFrame
    rating_data = pd.read_csv(rating_path, delimiter=',')
    book_data = pd.read_csv(book_path, delimiter=',')
    
    # copy dataframe book_data, and delete some feature.
    book_copy = book_data.copy()
    book_copy = book_copy.drop(columns=['best_book_id','work_id','books_count','isbn',
           'isbn13','title','language_code','average_rating',
           'ratings_count', 'work_ratings_count', 'work_text_reviews_count',
           'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
           'small_image_url'], axis=1)
    book_copy.head(3)
    
    # fill null values in book_data
    print("Missing values before fillna: ", book_copy.isnull().sum())
    book_copy['original_publication_year'] = book_copy['original_publication_year'].fillna(0)
    book_copy['original_title'] = book_copy['original_title'].fillna(book_data['title'])
    print("Missing values after fillna: ", book_copy.isnull().sum())
    
    # changes the data type original_publication_year column to int data type
    book_copy.loc[:, 'original_publication_year'] = book_copy['original_publication_year'].astype(int)
    book_copy.dtypes
    
    # drop duplicated rows
    print("Books shape before drop dup: ", book_copy.shape)
    print("Ratings shape before drop dup: ", rating_data.shape)
    book_copy.drop_duplicates(subset = ['book_id', 'goodreads_book_id'], inplace = True)
    rating_data.drop_duplicates(subset=['user_id','book_id'], inplace = True)
    print("Books shape after drop dup: ", book_copy.shape)
    print("Ratings shape after drop dup: ", rating_data.shape)
    
    return book_copy, rating_data

book_copy, rating_data = load_data(book_path, rating_path)

# Memisahkan 40 Genre -------------------------------------------------------------------

# Memisahkan 40 Genre

book_tags = pd.read_csv(book_tags_path, delimiter=',')
tags = pd.read_csv(tags_path, delimiter=',')

genres = ["Art", "Biography", "Business", "Chick Lit", "Children", "Christian", "Classics",
          "Comics", "Contemporary", "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction",
          "Gay and Lesbian", "Graphic Novels", "Historical Fiction", "History", "Horror",
          "Humor and Comedy", "Manga", "Memoir", "Music", "Mystery", "Nonfiction", "Paranormal",
          "Philosophy", "Poetry", "Psychology", "Religion", "Romance", "Science", "Science Fiction", 
          "Self Help", "Suspense", "Spirituality", "Sports", "Thriller", "Travel", "Young Adult"]

genres = list(map(str.lower, genres))

def create_genre_list(tag):
    """
    Function for building list of extracted genres
    """
    genre_list = []
    string_tag = str(tag)
    
    for genre in genres:
        
        if ('nonfiction' in string_tag):
            genre_list.append('nonfiction')
        elif (genre in string_tag) & ('non' not in string_tag):
            genre_list.append(genre)
        elif ('sci-fi' in string_tag) | ('scifi' in string_tag):
            genre_list.append('science fiction')
        else:
            pass
        
    return genre_list


def unique_array(list_):
    unique_list = list(set(list_))
    return unique_list

def extract_genres(book_tags, tags, genres):
    """
    Function to extract genres from tag names
    """
    tags['tag_name_lower'] = tags['tag_name'].str.lower()
    available_genres = tags.loc[tags.tag_name_lower.str.lower().isin(genres)]
    available_genres.head()
    
    tags['genre_list'] = [[]] * tags.shape[0]   

    # Add tags
    tags['genre_list'] = tags.apply(lambda row: create_genre_list(row['tag_name_lower']), axis = 1)
    tags_filtered = tags[tags.genre_list.str.len() != 0]
    
    # join with books
    booktags_to_genre = pd.merge(book_tags, tags_filtered, how = "left", on = "tag_id")
    booktags_to_genre.dropna(subset = ["genre_list"], inplace = True)
    booktags_to_genre.drop(['tag_id', 'tag_name', 'tag_name_lower', 'count'], axis=1, inplace = True)
    gr_book_genres = booktags_to_genre.groupby('goodreads_book_id').agg({'genre_list': 'sum'}).reset_index(drop = False)

    gr_book_genres['genres'] = gr_book_genres.apply(lambda row: unique_array(row['genre_list']), axis = 1)
    gr_book_genres.drop(['genre_list'], axis = 1, inplace = True)
    
    # Join with books
    books_with_genres = pd.merge(book_copy, gr_book_genres, how = "left", on = "goodreads_book_id")
    books_with_genres = books_with_genres[["book_id", "genres"]]
    
    return books_with_genres

books_with_genres = extract_genres(book_tags, tags, genres)

# ----------------------------------------------------------------------------------------------

# reduce data (make sampling)
book_id_limit = 2500
user_id_limit = 13356
book_data_small  = book_copy.drop(book_copy[book_copy['book_id'] > book_id_limit].copy().index)
rating_data_small = rating_data.loc[(rating_data['user_id'] <= user_id_limit) & (rating_data['book_id'] <= book_id_limit)]

pickle.dump(book_data_small,open('books.pkl','wb'))

# ----------------------------------------------------------------------------------------------

# Non Personalized

def popular_books(rating_data, book_data):
    #count the number of ratings given for each book and store the result in a new df called 'rating_count'
    rating_count = rating_data.groupby('book_id').count()['rating'].reset_index()
    rating_count.rename(columns={'rating':'rating_count'}, inplace=True)
    
    #count the mean of ratings given for each book and store the result in a new df called 'mean_rating'
    mean_rating = rating_data.groupby('book_id').mean().round(2)['rating'].reset_index()
    mean_rating.rename(columns={'rating':'mean_rating'}, inplace=True)
    
    #merge 'rating_count' dataframe with 'mean_rating' dataframe based on 'book_id' column
    popular = rating_count.merge(mean_rating, on='book_id')
    
    #merge df 'popular' with df 'book_copy' based on column 'book_id' then select specific columns and remove duplicate rows based on 'book_id'
    popular = popular.merge(book_data, on="book_id").drop_duplicates("book_id")[["book_id","rating_count","mean_rating","authors","original_publication_year","original_title","image_url"]]

    #merge df 'popular' with genres
    popular_with_genres = popular.merge(books_with_genres, on="book_id").drop_duplicates("book_id")[["book_id","rating_count","mean_rating","authors","original_publication_year","original_title","genres","image_url"]]
    
    return popular_with_genres

popular_with_genres = popular_books(rating_data_small, book_data_small)

#show the order of values from largest to smallest
top_30 = popular_with_genres.sort_values("rating_count", ascending=False).head(30)

pickle.dump(top_30, open('output/popular.pkl','wb'))
pickle.dump(popular_with_genres, open('output/popular_with_genres.pkl','wb'))

# ----------------------------------------------------------------------------------------------

# Personalized 

#load library
import surprise
from surprise import accuracy, Dataset, Reader, BaselineOnly, KNNBasic, KNNBaseline, SVD, NMF
from surprise.model_selection.search import RandomizedSearchCV
from surprise.model_selection import cross_validate, train_test_split

# Preparation Train and Test
#make matrix
#copy rating_data and make pivot to check total 'user_id' and 'book_id'
user_rating_pivot = rating_data_small.pivot(index='user_id',columns='book_id',values='rating')
#Initialize a Reader object in the Surprise library to read rating data on a scale of 1-5
reader = Reader(rating_scale = (1, 5))
#reads the rating data and converts it into a format that can be used to load the recommendation dataset from df 'rating_data'
dataset = Dataset.load_from_df(rating_data_small[['user_id', 'book_id', 'rating']].copy(), reader)
#split dataset into training data and test data
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
#validate splitting
print('Size train and test data:')
print(train_data.n_ratings, len(test_data))

# Train with KNN-Basic
#intialize ber hyperparams
best_params = pickle.load(open('output/best_params.pkl','rb'))
#create obj. and retrain whole train data
model_best = KNNBaseline(**best_params)
model_best.fit(train_data)
#predict test data using best model
test_pred = model_best.test(test_data)
test_rmse = accuracy.rmse(test_pred)

print("Test RMSE best model: ", str(test_rmse))


pickle.dump(model_refit, open('model/best_model.pkl','wb'))
print("Best model has been saved."
