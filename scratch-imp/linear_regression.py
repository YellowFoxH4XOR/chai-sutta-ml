import numpy as np

# add extra ones in each row starting for x0
X = np.array([
    [1,1,2,3,4],
    [1,5,6,7,8],
    [1,9,10,11,12],
    [1,13,14,15,16]
])

y = np.array([11, 22, 33, 44])

# calculating the normalized equation 
X_transpose = X.transpose()
X_transpose_dot_X = np.dot(X_transpose, X)
pinv = np.linalg.pinv(X_transpose_dot_X)
theta_pinv = np.dot(np.dot(pinv, X_transpose), y)

new_instance = np.array([1, 17, 18, 19, 20])
output = np.sum(np.dot(new_instance, theta_pinv))

assert int(output) == 55