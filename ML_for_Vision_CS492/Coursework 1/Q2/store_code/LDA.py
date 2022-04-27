import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
mat = scipy.io.loadmat('face.mat')
print(mat)
print(mat.keys())
# mat.keys() = dict_keys(['__header__', '__version__', '__globals__', 'X', 'l'])

X = mat['X'].T
l = mat['l'][0]

"""
X :  
[[106 130 191 ...  70 183  77]
 [103 129 185 ...  62 250  72]
 ...
 [ 36  36  19 ...  89  95 103]
 [ 41  35  17 ...  94 106 110]]

X.shape =  (2576, 520)
notice that the image is 46*56 = 2576 

l :  [ 1  1  1  1  1  1  1  1  1  1  2  2  2  2  2  2  2  2  2  2  3  3  3  3 ... 51 51 51 51 51 51 52 52 52 52 52 52 52 52 52 52]
list(set(l[0])) = [1, 2, 3, 4, 5, 6, 7, 8, ... 48, 49, 50, 51, 52] = list(range(1,53))

l.shape =  (520,)
"""


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

num_imgs_each_class = {}
for i in range(1, 53):
    num_imgs_each_class[i] = 0
for i in range(train_l.shape[0]):
    num_imgs_each_class[train_l[i]] += 1
print(num_imgs_each_class)
# num_imgs_each_class = {1: 8, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, 10: 8, 11: 8, 12: 8, 13: 8, 14: 8, 15: 8, 16: 8, 17: 8, 18: 8, 19: 8, 20: 8, 21: 8, 22: 8, 23: 8, 24: 8, 25: 8, 26: 8, 27: 8, 28: 8, 29: 8, 30: 8, 31: 8, 32: 8, 33: 8, 34: 8, 35: 8, 36: 8, 37: 8, 38: 8, 39: 8, 40: 8, 41: 8, 42: 8, 43: 8, 44: 8, 45: 8, 46: 8, 47: 8, 48: 8, 49: 8, 50: 8, 51: 8, 52: 8}

N = train_X.shape[0]
D = train_X.shape[1]
c = len(list(set(train_l)))

# c = 52

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
    # w:  [ 3.13349495e+05 -1.56241451e+03 -1.04590011e+03 ... -6.97437942e+00 2.41898479e+00 -7.55243986e-01]
    # w.shape =  (2576,)
    # v.shape =  (2576, 2576)
    pair_eigenvalue_eigenvector = []
    for i in range(w.shape[0]):
        pair_eigenvalue_eigenvector.append((w[i], v[:, i]))  # be careful
    decrease_order_pair_eigenvalue_eigenvector = sorted(pair_eigenvalue_eigenvector, key=lambda x: x[0], reverse=True)
    tmp = sorted(pair_eigenvalue_eigenvector, key=lambda x: x[0])  # [((-1.0657958809442602e-10+0j), array([-0.00193133+0.j,  0.00046664+0.j, -0.00713939+0.j, ..., ]

    # decrease_order_pair_eigenvalue_eigenvector[:10] = [((936388.6255239717+0j), array([-0.01228713+0.j,  ..., 0.00274912+0.j])), ((507218.2606004201+0j), array([-0.0192394 +0.j, ...,0.04521234+0.j])),... ((83675.7360421459+0j), array([-0.02647438+0.j, ...,0.05722392+0.j]))]

    return mean_face, decrease_order_pair_eigenvalue_eigenvector

def PCA_matrix(M_pca=400):
    mean_face, decrease_order_pair_eigenvalue_eigenvector = PCA(train_X, train_l)
    best_pair_eigenvalue_eigenvector = decrease_order_pair_eigenvalue_eigenvector[:M_pca]

    best_eigenvalue = np.array([ob[0] for ob in best_pair_eigenvalue_eigenvector])
    best_pca_eigenvector = np.array([ob[1] for ob in best_pair_eigenvalue_eigenvector])
    # best_eigenvalue.shape =  (464,)
    # best_eigenvector.shape =  (464, 2576)
    W_pca = np.zeros(shape=(M_pca, best_pca_eigenvector.shape[1]))
    # W_pca.shape = (M_pca, D)
    for i in range(M_pca):
        W_pca[i] = best_pca_eigenvector[i]
    return mean_face, W_pca


mean_face, W_pca = PCA_matrix()


def LDA_matrix(data, labels, M_lda):  # data of the size (n, M_pca), labels of the size (n,)
    original_dimension = data.shape[1]
    num_data_images = data.shape[0]
    number_images_each_class = {}
    mean_vector_each_class = {}
    different_labels = list(set(labels))
    num_classes = len(different_labels)

    sum_overall_vector = np.zeros(shape = data.shape[1])
    for i in range(data.shape[0]):
        sum_overall_vector += data[i]
    m = sum_overall_vector/num_data_images
    # m:  [122.72355769 122.24278846 120.34134615 ...  84.81490385  82.81009615 81.23798077]

    for i in range(num_classes):
        number_images_each_class[different_labels[i]] = 0
    for i in range(labels.shape[0]):
        number_images_each_class[labels[i]] += 1
    # number_images_each_class =  {1: 8, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, 10: 8, 11: 8, 12: 8, 13: 8, 14: 8, 15: 8, 16: 8, 17: 8, 18: 8, 19: 8, 20: 8, 21: 8, 22: 8, 23: 8, 24: 8, 25: 8, 26: 8, 27: 8, 28: 8, 29: 8, 30: 8, 31: 8, 32: 8, 33: 8, 34: 8, 35: 8, 36: 8, 37: 8, 38: 8, 39: 8, 40: 8, 41: 8, 42: 8, 43: 8, 44: 8, 45: 8, 46: 8, 47: 8, 48: 8, 49: 8, 50: 8, 51: 8, 52: 8}

    for i in range(num_classes):
        sum_i = np.zeros(shape=data[0].shape)
        label = different_labels[i]
        for j in range(num_data_images):
            if labels[j] == label:
                sum_i += data[j]
        mean_vector_each_class[label] = sum_i/number_images_each_class[label]

    # mean_vector_each_class:  {1: array([119.125, 119.5  , 120.5  , ...,  58.625,  58.5  ,  59.   ]), 2: array([134.25 , 140.125, 136.5  , ...,  59.875,  61.   ,  57.375]), 3: array([ 93.625,  98.75 , 101.375, ...,  93.375,  90.5  ,  71.   ]), 4: array([116.875, 117.   , 106.375, ...,  73.   ,  82.625,  82.625]), 5: array([141.75 , 142.5  , 143.875, ...,  58.25 ,  54.   ,  47.625]), 6: array([116.875, 108.75 ,  99.375, ...,  70.75 ,  69.   ,  69.   ]), 7: array([87.875, 90.5  , 98.125, ..., 98.375, 95.25 , 91.   ]), 8: array([128.625, 124.   , 118.375, ..., 105.   , 107.   , 112.   ]), 9: array([120.25 , 116.5  , 121.25 , ...,  99.875, 106.75 , 104.75 ]), 10: array([107.   , 107.75 , 105.375, ..., 103.125, 108.5  , 107.5  ]), 11: array([128.125, 128.375, 125.75 , ...,  70.625,  70.25 ,  66.5  ]), 12: array([109.5  , 107.5  , 105.625, ...,  74.375,  90.875, 126.75 ]), 13: array([111.75 , 108.375, 102.25 , ...,  53.   ,  52.25 ,  50.5  ]), 14: array([112.75 , 116.875, 118.375, ...,  92.625,  90.25 ,  89.375]), 15: array([136.75 , 128.75 , 120.625, ...,  88.   ,  89.   ,  95.375]), 16: array([102.875, 113.375, 121.   , ...,  69.5  ,  63.75 ,  63.375]), 17: array([102.75 , 108.25 , 105.375, ..., 117.625, 108.25 ,  92.   ]), 18: array([120.25 , 126.5  , 135.125, ...,  73.125,  71.25 ,  67.   ]), 19: array([113.625, 105.5  ,  99.25 , ...,  94.   ,  91.   ,  90.   ]), 20: array([132.25 , 130.25 , 128.625, ...,  66.75 ,  53.625,  50.5  ]), 21: array([107.5  , 106.75 , 105.625, ...,  53.375,  51.25 ,  51.125]), 22: array([137.5  , 137.875, 140.75 , ...,  98.125,  95.875,  95.75 ]), 23: array([143.125, 147.625, 148.125, ...,  58.125,  54.125,  51.625]), 24: array([111.125, 104.125,  99.5  , ...,  84.5  ,  82.875,  81.5  ]), 25: array([104.375, 104.375,  93.5  , ...,  60.75 ,  54.875,  51.375]), 26: array([131.375, 123.25 , 104.125, ..., 107.   , 105.   , 102.   ]), 27: array([113.375, 128.25 , 133.25 , ...,  44.875,  41.375,  42.375]), 28: array([151.375, 152.75 , 153.   , ...,  73.   ,  69.375,  67.   ]), 29: array([112.875, 109.5  , 105.625, ...,  82.875,  75.375,  70.   ]), 30: array([121.   , 111.375, 109.875, ..., 107.   ,  88.125,  85.875]), 31: array([111.75 , 111.625, 111.5  , ...,  85.875,  78.5  ,  71.875]), 32: array([126.   , 121.125, 121.125, ...,  50.125,  49.375,  50.125]), 33: array([131.75 , 128.125, 134.5  , ...,  82.   ,  94.875,  84.5  ]), 34: array([154.125, 157.25 , 144.5  , ...,  79.5  ,  82.   ,  81.   ]), 35: array([139.875, 130.   , 134.75 , ...,  93.   ,  90.   ,  90.875]), 36: array([138.5  , 127.75 , 114.25 , ..., 135.5  , 132.5  , 132.875]), 37: array([122.75 , 127.75 , 132.125, ..., 116.25 , 118.625, 104.875]), 38: array([149.375, 139.125, 131.25 , ...,  73.625,  74.75 ,  74.625]), 39: array([ 89.25 ,  87.5  ,  83.125, ..., 137.25 , 133.25 , 132.125]), 40: array([125.625, 116.875, 114.5  , ...,  83.25 ,  83.75 ,  80.75 ]), 41: array([121.875, 117.375, 107.25 , ...,  49.375,  46.   ,  43.625]), 42: array([128.   , 117.75 , 119.375, ..., 122.75 , 123.375, 116.25 ]), 43: array([132.5  , 130.   , 123.5  , ..., 102.5  ,  96.25 ,  91.125]), 44: array([138.375, 145.125, 149.125, ...,  57.75 ,  55.125,  52.625]), 45: array([121.375, 139.125, 132.   , ..., 103.5  , 104.125, 104.125]), 46: array([102.875, 112.625, 118.375, ...,  90.625,  87.375,  87.5  ]), 47: array([132.875, 136.375, 137.75 , ..., 130.25 , 122.125, 128.25 ]), 48: array([146.625, 148.125, 147.125, ..., 119.625, 101.75 ,  95.25 ]), 49: array([118.125, 121.   , 121.75 , ...,  67.   ,  62.875,  65.875]), 50: array([117.25 , 115.5  , 111.25 , ...,  97.875,  95.   ,  92.375]), 51: array([136.875, 138.75 , 137.875, ...,  62.625,  58.   ,  58.875]), 52: array([125.375, 122.75 , 124.25 , ...,  80.625,  84.625,  93.   ])}
    # number_images_each_class:  {1: 8, 2: 8, 3: 8, 4: 8, 5: 8, 6: 8, 7: 8, 8: 8, 9: 8, 10: 8, 11: 8, 12: 8, 13: 8, 14: 8, 15: 8, 16: 8, 17: 8, 18: 8, 19: 8, 20: 8, 21: 8, 22: 8, 23: 8, 24: 8, 25: 8, 26: 8, 27: 8, 28: 8, 29: 8, 30: 8, 31: 8, 32: 8, 33: 8, 34: 8, 35: 8, 36: 8, 37: 8, 38: 8, 39: 8, 40: 8, 41: 8, 42: 8, 43: 8, 44: 8, 45: 8, 46: 8, 47: 8, 48: 8, 49: 8, 50: 8, 51: 8, 52: 8}

    S_B = np.zeros(shape = (original_dimension, original_dimension))
    S_W = np.zeros(shape = (original_dimension, original_dimension))
    for i in range(num_classes):
        label = different_labels[i]
        m_i = mean_vector_each_class[label].copy()
        tmp1 = (m_i - m).reshape((original_dimension,1))
        tmp2 = tmp1.dot(tmp1.T)
        S_B += number_images_each_class[label]*tmp2
    # S_B.shape =  (original_dimension, original_dimension)
    # S_B.dtype =  float64

    for i in range(num_classes):
        label = different_labels[i]
        m_i = mean_vector_each_class[label].copy()

        tmp = np.zeros(shape = (original_dimension, original_dimension))
        for j in range(num_data_images):
            if labels[j] == label:
                x= data[j].copy()
                tmp1 = (x- m_i).reshape((original_dimension,1))
                # tmp1.dot(tmp1.T).shape = (original_dimension, original_dimension)
                tmp+= tmp1.dot(tmp1.T)
                # tmp.shape =  (2576, 2576) = (D, D)
        S_W += tmp
    # S_W.shape = (original_dimension, orignial_dimension)
    # S_B.shape = (original_dimension, orignial_dimension)
    # S_W.dtype =  float64
    SW_SB_matrix = (np.linalg.inv(S_W)).dot(S_B)
    w, v = np.linalg.eig(SW_SB_matrix)
    print("S_W: ", S_W)
    print("S_B: ", S_B)

    # w.shape = (original_dimension, )
    # v.shape = (original_dimension, original_dimension)

    pairs_eigenvalues_eigenvectors = [(w[i], v[:,i]) for i in range(w.shape[0])]
    decreasing_eigenvalues_eigenvectors = sorted(pairs_eigenvalues_eigenvectors, key= lambda x: x[0], reverse= True)
    best_eigenvalues_eigenvectors = decreasing_eigenvalues_eigenvectors[:M_lda]

    best_eigenvalues = [ob[0] for ob in best_eigenvalues_eigenvectors]
    best_eigenvectors = [ob[1] for ob in best_eigenvalues_eigenvectors]

    # best_eigenvalues:  [(3411093848770967.5+0j), (624784866760702.1+0j), (476779352062010.8+0j), (344893572043451.1+188189011397186.88j), (344893572043451.1-188189011397186.88j), (224754634390089.66+0j), (185558503424656+118208153648091.98j), (185558503424656-118208153648091.98j), (171648944045499.7+0j), (121975978150038.34+129731770098970.88j), (121975978150038.34-129731770098970.88j), (88258044929575.22+197832631777762.75j), (88258044929575.22-197832631777762.75j), (83152318342239.12+40991431346795.1j), (83152318342239.12-40991431346795.1j), (58792818986528.23+0j), (53874241847748.8+170283719599315.9j), (53874241847748.8-170283719599315.9j), (47467994423129.34+57384678038203.61j), (47467994423129.34-57384678038203.61j)]

    W_LDA = np.zeros(shape = (M_lda, original_dimension))
    for i in range(M_lda):
        tmp = best_eigenvectors[i].copy()
        tmp = tmp.astype('float64')
        W_LDA[i] = tmp

    # W_LDA.shape =  (M_lda, original_dimension)
    return W_LDA
W_LDA = LDA_matrix(train_X, train_l, M_lda = 51)
# W_LDA.shape =  (M_lda, D)
def LDA_transform(data): # data of the shape (num_img, D)
    return data.dot(W_LDA.T) # return shape (num_img, M_lda)

lda_train_X = LDA_transform(train_X)
# lda_train_X.shape =  (416, M_lda)
print("lda_train_X[:5] = ", lda_train_X[:5])
lda_test_X = LDA_transform(test_X)
print("lda_train_X.shape = ", lda_train_X.shape)
print("lda_train_X = ", lda_train_X)
# lda_test_X.shape = (num_img, M_lda)
def predict_test_X():
    l_preds = []
    for i in range(lda_test_X.shape[0]):
        x_i = lda_test_X[i]
        list_distance = []
        for j in range(lda_train_X.shape[0]):
            x_j = lda_train_X[j].copy()
            dist_i_j = np.linalg.norm(x_i - x_j)
            list_distance.append((train_l[j], dist_i_j))
        increasing_list_distance = sorted(list_distance, key= lambda x: x[1])
        l_preds.append(increasing_list_distance[0][0])

    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(l_preds, test_l)
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
    print("Overall accuracy : {}".format(acc * 100))
    return l_preds

l_preds = predict_test_X()



# lda_test_X[0] =  [ 18.19418466   6.51701327  -2.83743597  -0.94689525  -0.94689525 ... -3.41485538   3.58134107   3.58134107  -7.61244917  -7.61244917]













































