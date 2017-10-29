import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

print(scipy.stats.norm(100, 12).pdf(100))

print(scipy.stats.norm(100, 12).cdf(100))

print(scipy.stats.norm(100, 12).pdf(99))
print(scipy.stats.norm(100, 12).pdf(98))
print(scipy.stats.norm(100, 12).pdf(97))
print(scipy.stats.norm(100, 12).pdf(96))
print(scipy.stats.norm(100, 12).pdf(95))
print(scipy.stats.norm(100, 12).pdf(94))
print(scipy.stats.norm(100, 12).pdf(93))

df_iris = pd.read_csv('./iris.txt', sep=' ')

keys = ['sl', 'sw', 'pl', 'pw']

for i in range(3):
    k = i+1
    while k < 4:
        plt.scatter(df_iris[keys[i]], df_iris[keys[k]], c=df_iris['c'], cmap='prism')
        plt.xlabel(keys[i])
        plt.ylabel(keys[k])
        #plt.show()
        k += 1




sns.heatmap(df_iris.corr(), annot=True)
plt.show()