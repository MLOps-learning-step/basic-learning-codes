import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

weights = []
params = []

for c in np.arange(-5,5):
    lr =LogisticRegression(C=10.**c, random_state=1) #random_state is for reproducibility
    lr.fit(X_train,y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

weights = np.array(weights)
plt.plot(params, weights)
plt.xlabel("C")
plt.ylabel("Weights")
plt.title("Weights vs C")
plt.show()
    