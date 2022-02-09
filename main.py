import pandas as pd
import plotly.express as px
import csv
import matplotlib.pyplot as mt
from sklearn.linear_model import LogisticRegression
import numpy as np

data = pd.read_csv('data.csv')

velocity = data['Velocity'].tolist()

escaped = data['Escaped'].tolist()

#fig = px.scatter(x=velocity,y=escaped)
#fig.show()

X = np.reshape(velocity , (len(velocity) , 1))
Y = np.reshape(escaped , (len(escaped) , 1))

rl = LogisticRegression()
rl.fit(X,Y)

mt.figure()
mt.scatter(X.ravel() , Y , color='blue' , zorder=20)

def model(x):
    return 1/(1+ np.exp(-x))

X_test = np.linspace(0, 100, 200)

chance = model(X_test * rl.coef_ + rl.intercept_).ravel()

mt.plot(X_test , chance , color='k' , linewidth = 3 )

mt.axhline(y = 0 , color='k' , linestyle = '-')
mt.axhline(y = 1 , color='k' , linestyle = '-')
mt.axhline(y = 0.5 , color='k' , linestyle = ':')

mt.axvline(x = X_test[165] , color='b' , linestyle='--')

mt.ylabel('y')
mt.xlabel('X')
mt.xlim(75, 85)

mt.show()


userScore = float(input("Enter the velocity --> "))
chance = model(userScore * rl.coef_ + rl.intercept_).ravel()

if chance <= 0.01:
  print("It will not get escaped")
elif chance >= 1:
  print("It will get escaped!")
elif chance < 0.5:
  print("It might not get escaped")
else:
  print("It may get escaped")