from LinearRegression import LinearReg
import numpy as np
from pandas import DataFrame as df

X = np.array(list(range(30)))
# Set your equation
y = 3 * X + 6

data = np.hstack((X[:, np.newaxis], y[:, np.newaxis]))

data_df = df(data=data, index=[
    i for i in range(1, data.shape[0] + 1)], columns=['X', 'y'], dtype=float)
print(data_df)
print()

epochs = 1000
reg = LinearReg(step=.001, epochs=epochs, converge_at=10**-20)

history, converged = reg.fit(X, y, graph=True)
print('converged:', converged)
print()
print('final weights:', reg.weights)

# Set values to predict
pred = reg.predict([1, 2, 3])
print('predict:', pred)
