import pandas as pd
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy

u_fields = ['user_id', 'user_gender', 'user_age', 'user_occupation', 'user_zipcode']
pd_users = pd.read_table('./users.dat', sep='::', header=None, names=u_fields)
# print(pd_users.head())

r_fields = ['user_id','movie_id','movie_rating','timestamp']
pd_ratings = pd.read_table('./ratings.dat', sep='::', header=None, names=r_fields)
# print(pd_ratings.head())

merged_df = pd_ratings.pivot(index='user_id', columns='movie_id',values='movie_rating').fillna(0)

print(merged_df.head())

print(len(merged_df))

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    e = 0
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P, Q)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            print('Error is below threshold, breaking loop. error:', e)
            break
    print('Returning factorization results with error:', e)
    return P, Q.T


R = numpy.array(merged_df)

N = len(R)
M = len(R[0])
K = 1

P = numpy.random.rand(N, K)
Q = numpy.random.rand(M, K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)
# print(nP)
# print(nQ.T)
print(nR)


