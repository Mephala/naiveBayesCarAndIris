import numpy


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


# R = [
#     [5, 3, 0, 1],
#     [4, 0, 0, 1],
#     [1, 1, 0, 5],
#     [1, 0, 0, 4],
#     [0, 1, 5, 4],
# ]
#
# R = numpy.array(R)
#
# N = len(R)
# M = len(R[0])
# K = 2
#
# P = numpy.random.rand(N, K)
# Q = numpy.random.rand(M, K)
#
# nP, nQ = matrix_factorization(R, P, Q, K)
# nR = numpy.dot(nP, nQ.T)

# print(nP)
# print(nQ)
# print(R)
# print(P)
# print(Q)
#print(nR)

R = [
    [1,2,0],
    [2,4,6],
    [3,6,9]
]
# R = [
#     [1,2,3],
#     [2,4,6],
#     [3,6,9]
# ]
R = numpy.array(R)

N = len(R)
M = len(R[0])
K = 3

P = numpy.random.rand(N, K)
Q = numpy.random.rand(M, K)

nP, nQ = matrix_factorization(R, P, Q, K)
nR = numpy.dot(nP, nQ.T)
print(nP)
print(nQ.T)
print(nR)

m1 = 'star_wars'
m2 = 'eternal'
m3 = 'russian_institute'
m4 = 'lotr'
m5 = 'hitch'
m6 = 'carmen_electra'

R = [
    [5,1,5,0,0,0],
    [0,0,5,5,0,5],
    [5,0,0,5,1,0],
    [1,5,5,1,0,0],
    [0,5,0,0,5,0],
    [5,0,5,5,0,5],
    [0,5,0,0,5,0],
    [0,0,5,5,0,0]
]

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