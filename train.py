# For retrain model in full data
import pickle
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------

# Load  Dataset 
#load data from path
rating_path = 'data/ratings.csv'
book_path = 'data/books.csv'
book_tags_path = 'data/book_tags.csv'
tags_path = 'data/tags.csv'

def load_data(book_path, rating_path):
    """
    Load and preprocess book and rating data

    Parameters
    ----------
    book_path : str
        The file ppath of the book data CSV file
    rating_data : str
        The file path of the rating data CSV file

    Returns
    -------
    book_copy : DataFrame
        A Dataframe containing preprocessed book data
    rating_data : DataFrame
        A Dataframe containing preprocessed rating data

    - subsetting only the used columns
    - fill in missing values
    - drop duplicate rows
    """
        
    # reads the CSV file data and saves it as a DataFrame
    rating_data = pd.read_csv(rating_path, delimiter=',')
    book_data = pd.read_csv(book_path, delimiter=',')
    
    # copy dataframe book_data, and drop unnecessary columns
    book_copy = book_data.copy()
    book_copy = book_copy.drop(columns=['best_book_id','work_id','books_count','isbn',
           'isbn13','title','language_code','average_rating',
           'ratings_count', 'work_ratings_count', 'work_text_reviews_count',
           'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5',
           'small_image_url'], axis=1)
    book_copy.head(3)
    
    # fill missing values in book_copy DataFrame
    print("Missing values before fillna: ", book_copy.isnull().sum())
    book_copy['original_publication_year'] = book_copy['original_publication_year'].fillna(0)
    book_copy['original_title'] = book_copy['original_title'].fillna(book_data['title'])
    print("Missing values after fillna: ", book_copy.isnull().sum())
    
    # changes the data type original_publication_year column to int data type
    book_copy.loc[:, 'original_publication_year'] = book_copy['original_publication_year'].astype(int)
    book_copy.dtypes
    
    # drop duplicated rows in book_copy and rating_data
    print("Books shape before drop dup: ", book_copy.shape)
    print("Ratings shape before drop dup: ", rating_data.shape)
    book_copy.drop_duplicates(subset = ['book_id', 'goodreads_book_id'], inplace = True)
    rating_data.drop_duplicates(subset=['user_id','book_id'], inplace = True)
    print("Books shape after drop dup: ", book_copy.shape)
    print("Ratings shape after drop dup: ", rating_data.shape)
    
    return book_copy, rating_data

book_copy, rating_data = load_data(book_path, rating_path)

# ----------------------------------------------------------------------------------------------

# reduce data (make sampling)
book_id_limit = 2500
user_id_limit = 13356

# subset the book data and rating data to include only limited number of books, user and bookID
book_data_small  = book_copy.drop(book_copy[book_copy['book_id'] > book_id_limit].copy().index)
rating_data_small = rating_data.loc[(rating_data['user_id'] <= user_id_limit) & (rating_data['book_id'] <= book_id_limit)]

# save the small book data and rating data as pickle files
pickle.dump(book_data_small,open('output/books.pkl','wb'))
pickle.dump(rating_data_small,open('output/rating.pkl','wb'))

# --------------------------------------------------------------------------------------------

# separate 33 Genre
book_tags = pd.read_csv(book_tags_path, delimiter=',')
tags = pd.read_csv(tags_path, delimiter=',')

genres = ["Art", "Biography", "Business", "Children", "Christian", "Classics", "Comics",  
          "Contemporary", "Cookbooks", "Crime", "Ebooks", "Fantasy", "Fiction", "History", 
          "Horror", "Manga", "Memoir", "Music", "Mystery", "Nonfiction", "Paranormal", "Philosophy",
          "Poetry", "Psychology", "Religion", "Romance", "Science", "Science Fiction",   
          "Spirituality", "Sports", "Suspense", "Thriller", "Travel"]

genres = list(map(str.lower, genres))

def create_genre_list(tag):
    """
    Create a list of extracted genres based on input tags

    Parameters
    ----------
    tag : str
        A string containing tags or keyword
    
    Returns
    -------
    genre_list : list
        A list of genres extracted from the input tag
    """

    # initialize an empty list to store extracted genres
    genre_list = []
    # convert the input tag to a string
    string_tag = str(tag)
    
    # iterate through the predefined list of genres
    for genre in genres:
        
        # check if 'nonfiction' is explicitly mentioned in the tag
        if ('nonfiction' in string_tag):
            genre_list.append('nonfiction')
        # check if the genre is found in the tag and 'non' isn't part of the tag 
        elif (genre in string_tag) & ('non' not in string_tag):
            genre_list.append(genre)
        # check for variations of 'sci-fi' or 'scifi' in the tag and normalize to 'science fiction'
        elif ('sci-fi' in string_tag) | ('scifi' in string_tag):
            genre_list.append('science fiction')
        # if none of the above conditions are met, skip the genre
        else:
            pass
    # return the list of extracted genres    
    return genre_list


def unique_array(list_):
    """
    Create a new list containing unique elements from the input list

    Parameters
    ----------
    list_ : list
        The input list containing elements that may have duplicates

    Returns
    -------
    unique_list : list
        A new list with only unique elements from the input list
    """
    # use the 'set' data structure to remove duplicates from the input list
    unique_list = list(set(list_))
    # return the new list containing unique elements
    return unique_list

def extract_genres(book_tags, tags, genres):
    """
    Extract genres from tags and create DataFrame with book genres

    Parameters
    ----------
    book_tags : DataFrame
        DataFrame containing book tags
    tags : DataFrame
        DataFrame containing tag information
    genres : list
        List of genre names to be extracted

    Returns
    -------
    book_with_genres : DataFrame
        DataFrane with book genres
    
    This function takes book tags, tag info, and a list of genres as input
    It extracts relevant genre from the tags, associates them with books, and
    returns a DataFrame containing book IDs and their corresponding genres
    """
    # lowercase tag names for consistency
    tags['tag_name_lower'] = tags['tag_name'].str.lower()

    # filter tags that match the specified genres
    available_genres = tags.loc[tags.tag_name_lower.str.lower().isin(genres)]
    available_genres.head()    

    # initialize a 'genre_list' column in the 'tags' DataFrame
    tags['genre_list'] = [[]] * tags.shape[0]   

    # add tags to the 'genre_list' column based on genre extraction rules
    tags['genre_list'] = tags.apply(lambda row: create_genre_list(row['tag_name_lower']), axis = 1)
    
    # filter out tags with empty 'genre_list'
    tags_filtered = tags[tags.genre_list.str.len() != 0] 
    
    # join book tags with filtered tags based on 'tag_id'
    booktags_to_genre = pd.merge(book_tags, tags_filtered, how = "left", on = "tag_id")
    
    # drop rows with missing 'genre_list'
    booktags_to_genre.dropna(subset = ["genre_list"], inplace = True)
    
    # drop unnecessary columns from the merged DataFrame
    booktags_to_genre.drop(['tag_id', 'tag_name', 'tag_name_lower', 'count'], axis=1, inplace = True)
    
    # group book tags by 'goodreads_book_id' and aggregate genre list
    gr_book_genres = booktags_to_genre.groupby('goodreads_book_id').agg({'genre_list': 'sum'}).reset_index(drop = False)
    
    # create a 'genres' column by converting genre lists to unique arrays
    gr_book_genres['genres'] = gr_book_genres.apply(lambda row: unique_array(row['genre_list']), axis = 1)
    
    # drop the 'genre_list' column and join with the book data
    gr_book_genres.drop(['genre_list'], axis = 1, inplace = True)
    books_with_genres = pd.merge(book_data_small, gr_book_genres, how = "left", on = "goodreads_book_id")
    #books_with_genres = books_with_genres[["book_id", "original_title", "genres"]]
    
    return books_with_genres

# extract book genres and save the DataFrame as a pickle file
books_with_genres = extract_genres(book_tags, tags, genres)
print(books_with_genres)
print("Shape genre: ", books_with_genres.shape)
pickle.dump(books_with_genres,open('output/books_with_genres.pkl','wb'))

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
    popular_with_genres = popular.merge(books_with_genres, on="book_id").drop_duplicates("book_id")

    popular_with_genres['authors'] = popular_with_genres['authors_x'].fillna(popular_with_genres['authors_y'])
    popular_with_genres.drop(['authors_x', 'authors_y'], axis=1, inplace=True)

    popular_with_genres['original_publication_year'] = popular_with_genres['original_publication_year_x'].fillna(popular_with_genres['original_publication_year_y'])
    popular_with_genres.drop(['original_publication_year_x', 'original_publication_year_y'], axis=1, inplace=True)

    popular_with_genres['original_title'] = popular_with_genres['original_title_x'].fillna(popular_with_genres['original_title_y'])
    popular_with_genres.drop(['original_title_x', 'original_title_y'], axis=1, inplace=True)

    popular_with_genres['image_url'] = popular_with_genres['image_url_x'].fillna(popular_with_genres['image_url_y'])
    popular_with_genres.drop(['image_url_x', 'image_url_y'], axis=1, inplace=True)

    return popular_with_genres

# call the function to obtain popular books with genres
popular_with_genres = popular_books(rating_data_small, book_data_small)
#show the order of values from largest to smallest
top_30 = popular_with_genres.sort_values("rating_count", ascending=False).head(30)
print(top_30.head())
print("Shape popular: ", top_30.shape)

# save the top_30 as pickle files
pickle.dump(top_30, open('output/popular.pkl','wb'))

# ----------------------------------------------------------------------------------------------

# Personalized 
# load library
import surprise
from surprise import accuracy, Dataset, Reader, BaselineOnly, KNNBasic, KNNBaseline, SVD, NMF
from surprise.model_selection.search import RandomizedSearchCV
from surprise.model_selection import cross_validate, train_test_split

# Preparation Train and Test
# make matrix
# copy rating_data and make pivot to check total 'user_id' and 'book_id'
user_rating_pivot = rating_data_small.pivot(index='user_id',columns='book_id',values='rating')

# Initialize a Reader object in the Surprise library to read rating data on a scale of 1-5
reader = Reader(rating_scale = (1, 5))

# reads the rating data and converts it into a format that can be used to load the recommendation dataset from df 'rating_data'
dataset = Dataset.load_from_df(rating_data_small[['user_id', 'book_id', 'rating']].copy(), reader)

# split dataset into training data and test data
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# validate splitting
print('Size train and test data:')
print(train_data.n_ratings, len(test_data))

# Train with KNN-Basic
# intialize ber hyperparams
best_params = pickle.load(open('output/best_params.pkl','rb'))
#create obj. and retrain whole train data
model_best = KNNBaseline(**best_params)
model_best.fit(train_data)
#predict test data using best model
test_pred = model_best.test(test_data)
test_rmse = accuracy.rmse(test_pred)

print("Test RMSE best model: ", str(test_rmse))

# save the model_best as pickle files
pickle.dump(model_best, open('output/model_best.pkl','wb'))
print("Best model has been saved.")
