import numpy as np

class LogisticRegression:
    def __init__(self):
        self.b0, self.b1 = 0, 0
        self.sigmoid = np.array ([])

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
    
    # def predict (self, Xi):
    #     z = self.b0 + (self.b1 * Xi)
    #     sigmoid = 1 / (1 + np.exp (-z))
    #     if sigmoid >= 0.5:
    #         y_pred = 1
    #     else:
    #         y_pred = 0
    #     return sigmoid, y_pred

    def predict (self, Xi):
        z = self.b0 + (self.b1 * Xi)
        sigmoidFunction = [1/(1 + np.exp (-z))]
        self.sigmoid = np.append (self.sigmoid, sigmoidFunction)
        return self.sigmoid

if __name__ == '__main__':
    X = np.array ([
        [.50], [1.50], [2.00], [4.25], [3.25], [5.50]
    ])

    y = np.array ([
        0, 0, 0, 1, 1, 1
    ])

    model = LogisticRegression ()
    b0, b1 = model.fit (X=X, y=y)
    print (f'The value of intercept : {b0} \
           The value of slope : {b1}')
    
    sigmoid = model.predict (X)
    print (f'Sigmoid : {sigmoid}')

    result = np.array ([], dtype=int)
    for _ in range (len (sigmoid)):
        if sigmoid[_] >= 0.5:
            y_pred = 1
        else:
            y_pred = 0
        result = np.append (result, y_pred)

    print (f'True Label : {y}')
    print (f'Pred Label : {result}')
