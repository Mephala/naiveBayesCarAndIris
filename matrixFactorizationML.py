import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

u_fields = ['user_id', 'user_gender', 'user_age', 'user_occupation', 'user_zipcode']
pd_users = pd.read_table('./users.dat', sep='::', header=None, names=u_fields)
print(pd_users.head())

r_fields = ['user_id','movie_id','movie_rating','timestamp']
pd_ratings = pd.read_table('./ratings.dat', sep='::', header=None, names=r_fields)
print(pd_ratings.head())



