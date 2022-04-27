
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

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
print("test_l: ", test_l)
face_test_X = []
for i in range(test_X.shape[0]):
    face = test_X[i].reshape((46, 56)).T
    face_test_X.append(face)
face_test_X = np.array(face_test_X)
print(face_test_X.shape)
def save_test_data(data,label, namedataset = "test_imgs", img_extension = "png"):
    import os
    import glob
    import matplotlib.pyplot as plt
    if not os.path.isdir('images/' + namedataset):
        os.mkdir('images/' + namedataset)

    for i in range(data.shape[0]):
        img = data[i]
        class_img = label[i]
        if not os.path.isdir("images/" + namedataset + "/" + str(class_img)):
            os.mkdir("images/" + namedataset + "/" + str(class_img))
        paths_imgs = glob.glob("images/" + namedataset + "/" + str(class_img)  +"/*." + img_extension)
        index = len(paths_imgs)
        path = "images/" + namedataset + "/" + str(class_img) + "/" + str(index) + "." + img_extension
        plt.imsave(path, img)
#save_test_data(face_test_X, test_l)


def display_imgs(path_directory, img_extension, rows, columns, indexes_img_to_be_displayed = None):
    import glob
    paths = glob.glob(path_directory + '/*.' + img_extension)
    if indexes_img_to_be_displayed == None:
        num_imgs = len(paths)
        fig = plt.figure(figsize=(5, 2))
        for i in range(1, num_imgs + 1):
            img = plt.imread(paths[i - 1])
            a= fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            title = 'Class ' + path_directory.split('/')[-1]
            a.set_title(title)
        plt.show()
    else:
        particular_paths = [paths[i] for i in indexes_img_to_be_displayed]
        num_imgs = len(particular_paths)
        columns = num_imgs
        rows = 1
        fig = plt.figure(figsize=(5, 2))
        for i in range(1, num_imgs + 1):
            img = plt.imread(particular_paths[i - 1])
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
        plt.show()


display_imgs('images/predicted_imgs/PCA/fail/42', 'png', rows=1, columns= 2)



























































