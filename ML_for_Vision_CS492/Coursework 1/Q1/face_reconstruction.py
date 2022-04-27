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


mean_face, decrease_order_pair_eigenvalue_eigenvector = PCA(train_X, train_l)
reshape_mean_face = mean_face.copy()
reshape_mean_face = reshape_mean_face.reshape((46, 56)).T
# plt.imsave('save_imgs/mean_face/mean_face.jpg', reshape_mean_face)

# mean_face, decrease_order_pair_eigenvalue_eigenvector = low_dimension_PCA(X, l)
decreasing_eigenvalues = [ob[0] for ob in decrease_order_pair_eigenvalue_eigenvector]
decreasing_eigenvectors = np.array([ob[1] for ob in decrease_order_pair_eigenvalue_eigenvector], dtype = np.float64)
def display_1d_array():
    y = decreasing_eigenvalues
    x = [i for i in range(len(y))]
    plt.title('eigenvalues')
    plt.plot(x, y)
    plt.show()


def reconstruct_single_img(img, num_pca_base=100, h=46, w=56): # img is a 1D array

    best_pairs = decrease_order_pair_eigenvalue_eigenvector[:num_pca_base]
    best_decreasing_eigenvalues = [ob[0] for ob in best_pairs]
    best_decreasing_eigenvectors = np.array([ob[1] for ob in best_pairs])
    best_decreasing_eigenvectors = best_decreasing_eigenvectors.astype('float64')
    # X.shape =  (520, 2576)

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

    # plt.imshow(reconstructed_face_i.T)
    # plt.show()
    return reconstructed_face_i


#reconstruct_single_img(train_X[9], num_pca_base=200)


def reconstruct_X_dataset(dataset, listlabel, namedataset = "train_X", num_pca_base=100):
    import os
    import glob
    if not os.path.isdir('save_imgs/' + namedataset):
        os.mkdir('save_imgs/' + namedataset)

    os.mkdir('save_imgs/' + namedataset + '/' +str(num_pca_base))
    for label in list(set(listlabel)):
        os.mkdir('save_imgs/' + namedataset + '/' +str(num_pca_base) + '/' + str(label))
    for i in range(dataset.shape[0]):
        img = dataset[i]
        label = listlabel[i]
        reconstructed_face = reconstruct_single_img(img, num_pca_base)
        paths_in_directory = glob.glob('save_imgs/' + namedataset + '/' +str(num_pca_base) + '/' + str(label) + "/*.jpg")
        num_current_imgs_in_directory = len(paths_in_directory)
        index = num_current_imgs_in_directory
        path = 'save_imgs/' + namedataset + '/' +str(num_pca_base) + '/' + str(label) + '/' + str(index) + '_' + namedataset + '_' + str(num_pca_base) + '.jpg'
        plt.imsave(path, reconstructed_face)

def save_eigenfaces(namedataset = 'eigenface', num_eigenfaces= 10, h= 46, w = 56):
    import os
    import glob
    if not os.path.isdir('save_imgs/' + namedataset):
        os.mkdir('save_imgs/' + namedataset)
    for i in range(num_eigenfaces):
        x = decreasing_eigenvectors[i].reshape((h, w)).T
        paths_in_directory = glob.glob('save_imgs/' + namedataset + '/*.jpg' )
        index = len(paths_in_directory)
        path = 'save_imgs' + '/' + namedataset + '/' + str(index) + '.jpg'
        plt.imsave(path, x)

save_eigenfaces(namedataset='eigenface')



#reconstruct_X_dataset(train_X, train_l, namedataset= "train_X",num_pca_base= 50)
"""
reconstruct_X_dataset(X, l, namedataset= "X",num_pca_base= 25)
reconstruct_X_dataset(X, l, namedataset= "X",num_pca_base= 50)
reconstruct_X_dataset(X, l, namedataset= "X",num_pca_base= 100)
reconstruct_X_dataset(X, l, namedataset= "X",num_pca_base= 200)
reconstruct_X_dataset(X, l, namedataset= "X",num_pca_base= 400)

reconstruct_X_dataset(train_X, train_l, namedataset= "train_X",num_pca_base= 25)
reconstruct_X_dataset(train_X, train_l, namedataset= "train_X",num_pca_base= 50)
reconstruct_X_dataset(train_X, train_l, namedataset= "train_X",num_pca_base= 100)
reconstruct_X_dataset(train_X, train_l, namedataset= "train_X",num_pca_base= 200)
reconstruct_X_dataset(train_X, train_l, namedataset= "train_X",num_pca_base= 400)
"""

def display_multiple_img_from_directory(path_directory, img_extension, rows, columns, indexes_img_to_be_displayed = None):
    import glob
    paths = glob.glob(path_directory + '/*.' + img_extension)
    paths_combine_num_pca = []
    for i in range(len(paths)):
        path = paths[i]
        num_pca = int(path.split('.')[0].split('_')[-1])
        paths_combine_num_pca.append((path, num_pca))
    paths_combine_num_pca = sorted(paths_combine_num_pca, key= lambda x: x[1])
    all_paths = [ob[0] for ob in paths_combine_num_pca]
    # paths_combine_num_pca:  [('save_imgs/tam/0/original_0.jpg', 0), ('save_imgs/tam/0/0_X_25.jpg', 25), ('save_imgs/tam/0/0_X_50.jpg', 50), ('save_imgs/tam/0/0_X_100.jpg', 100), ('save_imgs/tam/0/0_X_200.jpg', 200), ('save_imgs/tam/0/0_X_400.jpg', 400)]
    paths_except_origin = [ob[0] for ob in paths_combine_num_pca if not 'original' in ob[0]]
    #paths_alone:  ['save_imgs/tam/0/0_X_25.jpg', 'save_imgs/tam/0/0_X_50.jpg', 'save_imgs/tam/0/0_X_100.jpg', 'save_imgs/tam/0/0_X_200.jpg', 'save_imgs/tam/0/0_X_400.jpg']

    if indexes_img_to_be_displayed == None:
        num_imgs = len(all_paths)
        fig = plt.figure(figsize=(10, 2))
        for i in range(1, num_imgs + 1):
            img = plt.imread(all_paths[i - 1])
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()
    else:
        num_imgs = len(paths_except_origin)
        columns = num_imgs
        rows = 1
        fig = plt.figure(figsize = (10,2))
        for i in range(1, num_imgs+1):
            img = plt.imread(paths_except_origin[i-1])
            name_img = paths_except_origin[i-1].split('/')[-1]
            title = 'num_pca = ' +  name_img.split('.')[0].split('_')[-1]
            print("title : ", name_img.split('.')[0].split('_')[-1])
            a = fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            a.set_title(title)
        plt.show()


#display_multiple_img_from_directory("save_imgs/train_X/50/1", "jpg", [0,1,3])

#display_multiple_img_from_directory("save_imgs/X/50/1", "jpg", rows = 1, columns= 10 , indexes_img_to_be_displayed = [0,1,3])

#display_multiple_img_from_directory("save_imgs/tam/48", "jpg", rows = 1, columns= 5 , indexes_img_to_be_displayed = [0,1,2,3,4])



def display_eigenface(path_directory, img_extension, rows, columns, indexes_img_to_be_displayed = None):
    import glob
    paths = glob.glob(path_directory + '/*.' + img_extension)
    if indexes_img_to_be_displayed == None:
        num_imgs = len(paths)
        fig = plt.figure(figsize=(10, 2))
        for i in range(1, num_imgs + 1):
            img = plt.imread(paths[i - 1])
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()
    else:
        particular_paths = [paths[i] for i in indexes_img_to_be_displayed]
        num_imgs = len(particular_paths)
        columns = num_imgs//2
        rows = 2
        fig = plt.figure(figsize=(10, 2))
        for i in range(1, num_imgs + 1):
            img = plt.imread(particular_paths[i - 1])
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()

display_eigenface('save_imgs/eigenface_low_dimension_PCA', 'jpg', rows = 2, columns= 5)


































