import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
mat = scipy.io.loadmat('face.mat')
print(mat)
print(mat.keys())
# mat.keys() = dict_keys(['__header__', '__version__', '__globals__', 'X', 'l'])

X = mat['X'].T
l = mat['l'][0]


def partion_train_test():
    train_X = []
    train_l = []

    test_X = []
    test_l = []
    train_index = [0, 1, 2, 3, 4, 5, 6, 7]
    test_index = [8, 9]

    for i in range(X.shape[0]):
        i_remainder_10 = i % 10
        if i_remainder_10 in train_index:
            train_X.append(X[i])
            train_l.append(l[i])
        else:
            test_X.append(X[i])
            test_l.append(l[i])
    return np.array(train_X), np.array(train_l), np.array(test_X), np.array(test_l)


train_X, train_l, test_X, test_l = partion_train_test()

"""
X:  [[106 103 103 ...  37  36  41]
 [130 129 130 ...  40  36  35]
 ...
 [183 250 163 ... 102  95 106]
 [ 77  72  68 ...  93 103 110]]
 X.shape =  (520, 2576)


 l:  [[ 1  1  1  1  1  1  1  1  1  1  2  2  2  2  2  2  2  2  2  2  3  3  3  3 ... 51 51 51 51 51 51 52 52 52 52 52 52 52 52 52 52]]
 l.shape =  (1, 520)


 train_X:  [[106 103 103 ...  37  36  41]
 [130 129 130 ...  40  36  35]
 ...
 [ 60  68  71 ...  41  48  65]
 [ 70  62  83 ...  87  89  94]]
 train_X.shape =  (416, 2576)


 train_l:  [ 1  1  1  1  1  1  1  1  2  2  2  2  2  2  2  2  3  3  3  3  3  3  3  3 ... 52 52 52 52 52 52 52 52]
 train_l.shape =  (416,)


 test_X:  
 [[ 96  93  95 ... 100  85 121]
 [ 93  95  81 ...  88 100  91]
 ...
 [183 250 163 ... 102  95 106]
 [ 77  72  68 ...  93 103 110]]
 test_X.shape =  (104, 2576)


 test_l:  [ 1  1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  9  9 10 10 11 11 12 12 ... 49 49 50 50 51 51 52 52]
 test_l.shape =  (104,)
"""


def PCA(X_data, labels):  # for example, X_data.shape = (416, 2576), labels.shape = (416,)
    num_images = X_data.shape[0]
    D = X_data.shape[1]
    list_classes = list(set(labels))  # [1, 2, 3, 4, 5, 6, 7, 8, ... 48, 49, 50, 51, 52]
    num_classes = len(list_classes)

    sum_face = np.zeros(shape=X_data.shape[1])
    for i in range(num_images):
        sum_face += X_data[i]
    mean_face = sum_face / num_images

    # step 4 and 5

    S = np.zeros((D, D))
    for i in range(num_images):  # (x-mc).dot((x-mc).T)
        phi_xn = X_data[i] - mean_face
        xn = phi_xn.reshape(D, 1)
        m = xn.dot(xn.T)
        # m.shape = (2576, 2576)
        S += m
    S = S / num_images
    w, v = np.linalg.eig(S)
    print("w: ", w)
    # w:  [ 3.13349495e+05 -1.56241451e+03 -1.04590011e+03 ... -6.97437942e+00 2.41898479e+00 -7.55243986e-01]
    # w.shape =  (2576,)
    # v.shape =  (2576, 2576)
    pair_eigenvalue_eigenvector = []
    for i in range(w.shape[0]):
        pair_eigenvalue_eigenvector.append((w[i], v[:, i]))  # be careful
    decrease_order_pair_eigenvalue_eigenvector = sorted(pair_eigenvalue_eigenvector, key=lambda x: x[0], reverse=True)
    tmp = sorted(pair_eigenvalue_eigenvector, key=lambda x: x[
        0])  # [((-1.0657958809442602e-10+0j), array([-0.00193133+0.j,  0.00046664+0.j, -0.00713939+0.j, ..., ]

    print(decrease_order_pair_eigenvalue_eigenvector[:10])
    # decrease_order_pair_eigenvalue_eigenvector[:10] = [((936388.6255239717+0j), array([-0.01228713+0.j,  ..., 0.00274912+0.j])), ((507218.2606004201+0j), array([-0.0192394 +0.j, ...,0.04521234+0.j])),... ((83675.7360421459+0j), array([-0.02647438+0.j, ...,0.05722392+0.j]))]

    return mean_face, decrease_order_pair_eigenvalue_eigenvector


def low_dimension_PCA(X_data, labels):
    num_images = X_data.shape[0]
    D = X_data.shape[1]
    list_classes = list(set(labels))  # [1, 2, 3, 4, 5, 6, 7, 8, ... 48, 49, 50, 51, 52]
    num_classes = len(list_classes)

    sum_face = np.zeros(shape=X_data.shape[1])
    for i in range(num_images):
        sum_face += X_data[i]
    mean_face = sum_face / num_images
    A = np.zeros(shape=(D, num_images))
    for i in range(num_images):
        phi_n = X_data[i] - mean_face
        A[:, i] = phi_n
    # A.shape =  (2576, 416)
    S = (A.T).dot(A)
    S = S / num_images

    # S.shape =  (416, 416)

    w, v = np.linalg.eig(S)
    # w.shape =  (416,)

    # v.shape =  (416, 416)

    pair_eigenvalue_eigenvector = []
    for i in range(w.shape[0]):
        pair_eigenvalue_eigenvector.append((w[i], v[:, i]))  # be careful
    decrease_order_pair_eigenvalue_eigenvector = sorted(pair_eigenvalue_eigenvector, key=lambda x: x[0], reverse=True)
    # decreasing_eigenvalues =  [936388.6255239721, 507218.26060042036, 478365.3988530388, 267221.967260991, 215489.7952065648, 148960.41780029793, 125026.35171848646, 110178.65523240414, 90206.8877427383, 83675.73604214602]
    tmp = sorted(pair_eigenvalue_eigenvector, key=lambda x: x[0])

    return mean_face, decrease_order_pair_eigenvalue_eigenvector

t0 = time.time()
mean_face, decrease_order_pair_eigenvalue_eigenvector = low_dimension_PCA(train_X, train_l)
t1 = time.time()
print("time to run is {} seconds".format(t1-t0))

decreasing_eigenvalues = [ob[0] for ob in decrease_order_pair_eigenvalue_eigenvector]

decreasing_eigenvectors = [ob[1] for ob in decrease_order_pair_eigenvalue_eigenvector]
decreasing_eigenvectors = np.array(decreasing_eigenvectors).astype('float64')

abs_decreasing_eigenvalues = [abs(ob) for ob in decreasing_eigenvalues]
abs_decreasing_eigenvalues = list(sorted(abs_decreasing_eigenvalues, reverse= True))

first20_eigenvalues = np.array(decreasing_eigenvalues[:20]).astype("float64")
print("first 20 eigenvalues : ", list(first20_eigenvalues))
print("first 20 eigenvectors : ", list(decreasing_eigenvectors[:20]))

first_zero = 0
while ( abs_decreasing_eigenvalues[first_zero]> 0.01):
    first_zero+=1
print(len(abs_decreasing_eigenvalues))
print("first_zero: ", first_zero)
print("zero eigenvalues : ", abs_decreasing_eigenvalues[first_zero-1:])
print("nonzero eigenvalues : ", abs_decreasing_eigenvalues[first_zero-10: first_zero+1])


def display_1d_array():
    y = decreasing_eigenvalues
    x = [i for i in range(len(y))]
    plt.title('eigenvalues')
    plt.plot(x, y)
    plt.show()

display_1d_array()




