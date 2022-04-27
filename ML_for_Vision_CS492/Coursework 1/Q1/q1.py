import scipy.io
import numpy as np
import matplotlib.pyplot as plt
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
    train_index = [0,1,2,3,4,5,6,7]
    test_index = [8,9]

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


def PCA(X_data, labels): # for example, X_data.shape = (416, 2576), labels.shape = (416,)
    num_images = X_data.shape[0]
    D = X_data.shape[1]
    list_classes = list(set(labels))  # [1, 2, 3, 4, 5, 6, 7, 8, ... 48, 49, 50, 51, 52]
    num_classes = len(list_classes)

    sum_face = np.zeros(shape = X_data.shape[1])
    for i in range(num_images):
        sum_face += X_data[i]
    mean_face = sum_face/num_images

    # step 4 and 5

    S = np.zeros((D,D))
    for i in range(num_images): # (x-mc).dot((x-mc).T)
        phi_xn = X_data[i] - mean_face
        xn = phi_xn.reshape(D,1)
        m = xn.dot(xn.T)
        # m.shape = (2576, 2576)
        S += m
    S = S/num_images
    w, v = np.linalg.eig(S)
    print("w: ", w)
    # w:  [ 3.13349495e+05 -1.56241451e+03 -1.04590011e+03 ... -6.97437942e+00 2.41898479e+00 -7.55243986e-01]
    # w.shape =  (2576,)
    # v.shape =  (2576, 2576)
    pair_eigenvalue_eigenvector = []
    for i in range(w.shape[0]):
        pair_eigenvalue_eigenvector.append((w[i], v[:,i]))   # be careful
    decrease_order_pair_eigenvalue_eigenvector = sorted(pair_eigenvalue_eigenvector, key = lambda x: x[0], reverse = True)
    tmp = sorted(pair_eigenvalue_eigenvector, key = lambda x: x[0]) # [((-1.0657958809442602e-10+0j), array([-0.00193133+0.j,  0.00046664+0.j, -0.00713939+0.j, ..., ]

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
    mean_face = sum_face/num_images
    A= np.zeros(shape = (D, num_images))
    for i in range(num_images):
        phi_n = X_data[i] - mean_face
        A[:,i] = phi_n.copy()
    # A.shape =  (2576, 416)
    S = (A.T).dot(A)
    S = S/num_images

    # S.shape =  (416, 416)

    w, v = np.linalg.eig(S)
    # w.shape =  (416,)

    # v.shape =  (416, 416)

    pair_eigenvalue_eigenvector = []
    for i in range(w.shape[0]):
        pair_eigenvalue_eigenvector.append((w[i], v[:, i]))  # be careful
    decrease_order_pair_eigenvalue_eigenvector = sorted(pair_eigenvalue_eigenvector, key=lambda x: x[0], reverse=True)
    low_D_dimension_eigenvectors = []
    for i in range(len(decrease_order_pair_eigenvalue_eigenvector)):
        z = decrease_order_pair_eigenvalue_eigenvector[i][1]
        z1 = A.dot(z)
        z2 = z1/np.linalg.norm(z1)
        low_D_dimension_eigenvectors.append(z2)
    tmp = sorted(pair_eigenvalue_eigenvector, key=lambda x: x[0])

    return low_D_dimension_eigenvectors, mean_face, decrease_order_pair_eigenvalue_eigenvector


#mean_face, decrease_order_pair_eigenvalue_eigenvector = PCA(X, l)
#reshape_mean_face = mean_face.copy()
#reshape_mean_face = reshape_mean_face.reshape((46,56)).T
#plt.imsave('save_imgs/mean_face/mean_face.jpg', reshape_mean_face)

def tmp_low_PCA():
    _, decrease_order_pair_eigenvalue_eigenvector = low_dimension_PCA(train_X, train_l)
    decreasing_eigenvalues = [ob[0] for ob in decrease_order_pair_eigenvalue_eigenvector]
    print("tam: ", decreasing_eigenvalues[413:416])
    print("top 20 eigenvalues: ", decreasing_eigenvalues[:20])
    decreasing_eigenvectors = [ob[1] for ob in decrease_order_pair_eigenvalue_eigenvector]
    print("top 20 eigenvectors: ", decreasing_eigenvectors[:20])

low_D_dimension_eigenvectors, low_mean_face , low_decrease_order_pair_eigenvalue_eigenvector = low_dimension_PCA(train_X, train_l)

low_decreasing_eigenvalues = [ob[0] for ob in low_decrease_order_pair_eigenvalue_eigenvector]

low_decreasing_eigenvectors = [ob[1] for ob in low_decrease_order_pair_eigenvalue_eigenvector]
print("low decreasing dimension: ", low_decreasing_eigenvectors[0].shape)
mean_face, decrease_order_pair_eigenvalue_eigenvector = PCA(train_X, train_l)
decreasing_eigenvalues = [ob[0] for ob in decrease_order_pair_eigenvalue_eigenvector]
decreasing_eigenvectors = np.array([ob[1] for ob in decrease_order_pair_eigenvalue_eigenvector]).astype('float64')
print("low D : ", low_D_dimension_eigenvectors[0])
print("normal: ", decreasing_eigenvectors[0])
print("magintude normal: ", np.linalg.norm(decreasing_eigenvectors[0]))
print("low D dimension: ", low_D_dimension_eigenvectors[0].shape)
print("normal dimension: ", decreasing_eigenvectors[0].shape)
print("low eigenvalues: ", low_decreasing_eigenvalues)
print("eigenvalues: ", decreasing_eigenvalues)

def tmp_PCA():
    _, decrease_order_pair_eigenvalue_eigenvector = PCA(train_X, train_l)

    decreasing_eigenvalues = [ob[0] for ob in decrease_order_pair_eigenvalue_eigenvector]
    decreasing_eigenvectors = [ob[1] for ob in decrease_order_pair_eigenvalue_eigenvector]
    decreasing_eigenvectors = list(np.array(decreasing_eigenvectors, dtype=np.float64))
    print("decreasing_eigenvalues: ", decreasing_eigenvalues[:20])
    print("decreasing_eigenvectors: ", decreasing_eigenvectors[:20])
    a0 = decreasing_eigenvalues[0]
    a10 = decreasing_eigenvalues[10]
    a50 = decreasing_eigenvalues[50]
    a100 = decreasing_eigenvalues[100]
    a400 = decreasing_eigenvalues[400]
    print("a0/a10: ", a0 / a10)
    print("a0/a50: ", a0 / a50)
    print("a0/a100: ", a0 / a100)
    print("a0/a400: ", a0 / a400)


def display_1d_array():
    import matplotlib
    font = {'size': 16}
    matplotlib.rc('font', **font)
    y = decreasing_eigenvalues[:20]
    x= [i for i in range(len(y))]
    plt.title('eigenvalues')
    plt.plot(x, y)
    plt.show()

#display_1d_array()

def reconstruct_single_img(img,num_pca_base = 100, h =46, w = 56):
    #img = X[index_img]

    best_pairs = decrease_order_pair_eigenvalue_eigenvector[:num_pca_base]
    best_decreasing_eigenvalues = [ob[0] for ob in best_pairs]
    best_decreasing_eigenvectors = np.array([ob[1] for ob in best_pairs])
    best_decreasing_eigenvectors = best_decreasing_eigenvectors.astype('float64')
    # X.shape =  (520, 2576)
    num_images = X.shape[0]
    D = X.shape[1]

    w_n = np.zeros(shape=(num_pca_base, 1))
    phi_xn = img - mean_face

    for j in range(num_pca_base):
        a_n_j = (phi_xn.T).dot(best_decreasing_eigenvectors[j])
        # a_n_j =  (-115.43420349189859+0j)
        w_n[j] = a_n_j
    # type(w_n) =  float64

    reconstructed_face_i = mean_face
    for k in range(num_pca_base):
        reconstructed_face_i += w_n[k] * best_decreasing_eigenvectors[k]

    # reconstructed_face_i.shape =  (2576,)
    reconstructed_face_i = reconstructed_face_i.reshape((h, w)).T

    #plt.imshow(reconstructed_face_i.T)
    #plt.show()
    return reconstructed_face_i

#reconstruct_single_img(9,num_pca_base=200)

def reconstruct_X_dataset(num_pca_base = 100):
    import os
    list_label = list(set(l))
    os.mkdir('save_imgs/'+ str(num_pca_base))
    for label in list_label:
        os.mkdir('save_imgs/' +str(num_pca_base) + '/' +str(label))
    for i in range(X.shape[0]):
        img = X[i]
        label = l[i]
        reconstructed_face = reconstruct_single_img(img, num_pca_base)
        index = i % 10
        path = 'save_imgs/' + str(num_pca_base) + '/' + str(label) + '/' + str(index) + '.jpg'
        plt.imsave(path, reconstructed_face)

#reconstruct_X_dataset()

def display_multiple_img_from_directory(path_directory, img_extension): # img_extension is jpg or png
    import glob
    paths = glob.glob(path_directory+ '/*.' + img_extension)
    num_imgs = len(paths)
    columns = num_imgs
    rows = 1
    fig = plt.figure(figsize= (5,10))
    for i in range(1, num_imgs+1):
        img = plt.imread(paths[i-1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()













































