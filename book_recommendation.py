from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from typing import List
import heapq
import csv
# link to the datasets ->> https://github.com/zygmuntz/goodbooks-10k
users = pd.read_csv("ratings.csv")
books = pd.read_csv("books.csv")
books = books[['book_id', 'average_rating', 'title']]


class CosineBookRecommender:

    def __init__(self, user_ratings: List, user_books: List):
        self.user_ratings = user_ratings
        self.user_books = user_books

    def load_data(self):
        user_ratings = pd.read_csv("ratings.csv")
        books_list = pd.read_csv("books.csv")
        books = books_list[['book_id', 'average_rating', 'title']]
        joint_matrix = pd.merge(user_ratings, books, on="book_id", how='left')
        rating_matrix = joint_matrix[['user_id', 'book_id', 'rating']]
        pivoted_matrix = rating_matrix.pivot(index='user_id', columns='book_id', values='rating')
        return pivoted_matrix

    def centered_ratings(self):
        ratings_users = self.load_data()
        ratings_normalised = ratings_users.sub(ratings_users.mean(axis=1), axis=0)
        return ratings_normalised

    def cosine_similarity(self):
        cosine_sim = dict()
        normalised_data = self.centered_ratings()
        # creating new user dataframe
        empty_df = pd.DataFrame(columns=normalised_data.columns)
        user_rating = np.array(self.user_ratings).reshape(1, len(self.user_ratings))

        new_user = pd.DataFrame(user_rating, columns=self.user_books)
        new_user_norm = new_user.sub(new_user.mean(axis=1), axis=0)
        new_user_adj_shape = new_user_norm.append(empty_df)
        new_user_adj_shape = new_user_adj_shape.replace(np.nan, 0.0, regex=True)

        normalised_data = normalised_data.replace(np.nan, 0.0, regex=True)

        for idx, row in normalised_data.iterrows():
            val = cosine_similarity(row.values.reshape(1, -1), new_user_adj_shape)
            if val > 0.0:
                print("Iteration:", idx)
                cosine_sim[row.name] = float(val)
        return {k: v for k, v in sorted(cosine_sim.items(), key=lambda item: item[1])}


with open('sample_user_rating.csv', 'r') as f:
    lines = f.readlines()
    user_rating = []
    user_books = []
    for line in lines[1:]:
        stripped_line = line.strip().split(',')
        user_rating.append(int(stripped_line[1]))
        user_books.append(int(stripped_line[0]))

recommender = CosineBookRecommender(user_rating, user_books)
res = recommender.cosine_similarity()
res = [(v, k) for k, v in res.items()]
most_correlated = heapq.nlargest(5, res)

final_dict = dict()

for _, user in most_correlated:
    user_books = users.loc[(users.user_id == user) & (users['rating'] == 5)]
    user_books_merged = pd.merge(user_books, books, on="book_id", how='left')
    user_books_merged = user_books_merged.loc[user_books_merged['average_rating'] > 4.2]
    user_books_merged = user_books_merged[['book_id', 'title', 'average_rating']]
    final_dict.update(user_books_merged.set_index('book_id').T.to_dict('list'))

for key in final_dict.keys():
    if key in user_books:
        final_dict.pop(key)


with open('sample_user_recommendations.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.writer(f)
    for key, value in final_dict.items():
        value.append(key)
        w.writerow(value)







