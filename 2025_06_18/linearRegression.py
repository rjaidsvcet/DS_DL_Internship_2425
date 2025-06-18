import numpy as np
from sklearn.linear_model import LinearRegression

class LinearRegression:
    def __init__(self):
        self.b0, self.b1 = 0, 0

    def fit (self, X, y):
        X_mean = np.mean (X)
        y_mean = np.mean (y)
        ssxy, ssx = 0, 0
        for _ in range (len (X)):
            ssxy += (X[_]-X_mean)*(y[_]-y_mean)
            ssx += (X[_]-X_mean)**2
        self.b1 = ssxy / ssx
        self.b0 = y_mean - (self.b1 * X_mean)
        return self.b0, self.b1
    
    def predict (self, Xi):
        y_hat = self.b0 + (self.b1 * Xi)
        return y_hat

if __name__ == '__main__':
    heights = np.array ([
        [160], [171], [182], [180], [154]
    ])

    weights = np.array ([
        72, 76, 77, 83, 76
    ])

    # print (f'The shape of X : {heights.shape} \
        #    and the shape of y : {weights.shape}')

    lr = LinearRegression ()
    b0, b1 = lr.fit (X=heights, y=weights)
    print (f'The value of intercept : {b0} \
           The value of slope : {b1}')
    
    Xi = [[176]]
    y_hat = lr.predict (Xi)
    print (f'The weight of the person with the \
           height of {Xi} is predicted as {y_hat}')

    # Sklearn Model
    # model = LinearRegression ()
    # model.fit (heights, weights)
    # y_pred = model.predict (Xi)
    # print (y_pred)