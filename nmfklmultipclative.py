import pandas as pd

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('data/ml-100k/u.data', sep='\t', header=None, names=rnames)
# ratings

inames = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
          'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
          'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi',
          'Thriller', 'War', 'Western']
items = pd.read_table('data/ml-100k/u.item', sep='|', header=None, names=inames)
# items

unames = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('data/ml-100k/u.user', sep='|', header=None, names=unames)
print(users)
