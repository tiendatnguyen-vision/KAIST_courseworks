
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('face.mat')
print(mat)
print(mat.keys())
# mat.keys() = dict_keys(['__header__', '__version__', '__globals__', 'X', 'l'])

X = mat['X'].T
l = mat['l'][0]

def storage_original_face():
    import os
    import glob
    if not os.path.isdir("save_imgs/original_face"):
        os.mkdir("save_imgs/original_face")
    distinct_label = list(set(l))
    for i in range(len(distinct_label)):
        label = str(distinct_label[i])
        os.mkdir("save_imgs/original_face/" + label)

    num_imgs = X.shape[0]
    for i in range(num_imgs):
        img = X[i].reshape((46,56)).T
        label = str(l[i])
        current_paths = glob.glob("save_imgs/original_face/" + label + "/*.jpg")
        index = len(current_paths)
        plt.imsave('save_imgs/original_face/' + label + '/' + "original_" +str(index) + '.jpg', img)

storage_original_face()

























































