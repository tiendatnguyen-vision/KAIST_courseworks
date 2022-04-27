import scipy.io
import numpy as np
import time
# mat.keys() = dict_keys(['__header__', '__version__', '__globals__', 'X', 'l'])

X = np.random.rand(300,100)*256
l = []
for i in range(1,31):
    for j in range(10):
        l.append(i)
l = np.array(l, dtype=np.uint8)


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

def partion_4_subbet():
    sub_trainX_1 = []
    sub_trainX_2 = []
    sub_trainX_3 = []
    sub_trainX_4 = []

    sub_train_l_1 = []
    sub_train_l_2 = []
    sub_train_l_3 = []
    sub_train_l_4 = []

    num_imgs = train_X.shape[0]
    index1 = [0,1]
    index2= [2,3]
    index3 = [4,5]
    index4 = [6,7]
    for i in range(num_imgs):
        remainer_i = i % 8
        if remainer_i in index1:
            sub_trainX_1.append(train_X[i])
            sub_train_l_1.append(train_l[i])
        elif remainer_i in index2:
            sub_trainX_2.append(train_X[i])
            sub_train_l_2.append(train_l[i])
        elif remainer_i in index3:
            sub_trainX_3.append(train_X[i])
            sub_train_l_3.append(train_l[i])
        else:
            sub_trainX_4.append(train_X[i])
            sub_train_l_4.append(train_l[i])
    return np.array(sub_trainX_1), np.array(sub_train_l_1), np.array(sub_trainX_2), np.array(sub_train_l_2), np.array(sub_trainX_3), np.array(sub_train_l_3), np.array(sub_trainX_4), np.array(sub_train_l_4)

sub_trainX_1, sub_train_l_1, sub_trainX_2, sub_train_l_2, sub_trainX_3, sub_train_l_3, sub_trainX_4, sub_train_l_4 = partion_4_subbet()
N1 = sub_trainX_1.shape[0]
N2 = sub_trainX_2.shape[0]
N3 = sub_trainX_3.shape[0]
N4 = sub_trainX_4.shape[0]

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
    a0 = time.time()
    w, v = np.linalg.eig(S)
    a1 = time.time()
    # w:  [ 3.13349495e+05 -1.56241451e+03 -1.04590011e+03 ... -6.97437942e+00 2.41898479e+00 -7.55243986e-01]
    # w.shape =  (2576,)
    # v.shape =  (2576, 2576)
    pair_eigenvalue_eigenvector = []
    for i in range(w.shape[0]):
        pair_eigenvalue_eigenvector.append((w[i], v[:, i]))  # be careful
    decrease_order_pair_eigenvalue_eigenvector = sorted(pair_eigenvalue_eigenvector, key=lambda x: x[0], reverse=True)

    # decrease_order_pair_eigenvalue_eigenvector[:10] = [((936388.6255239717+0j), array([-0.01228713+0.j,  ..., 0.00274912+0.j])), ((507218.2606004201+0j), array([-0.0192394 +0.j, ...,0.04521234+0.j])),... ((83675.7360421459+0j), array([-0.02647438+0.j, ...,0.05722392+0.j]))]

    return mean_face, S,  decrease_order_pair_eigenvalue_eigenvector

def PCA_matrix(data, label, M_pca): # for example : X = sub_trainX_1, l = sub_train_l_1
    t0 = time.time()
    mean_face, S, decrease_order_pair_eigenvalue_eigenvector = PCA(data, label)
    t1 = time.time()

    best_pair_eigenvalue_eigenvector = decrease_order_pair_eigenvalue_eigenvector[:M_pca]

    best_eigenvalue = np.array([ob[0] for ob in best_pair_eigenvalue_eigenvector])
    best_pca_eigenvector = np.array([ob[1] for ob in best_pair_eigenvalue_eigenvector])
    # best_eigenvalue.shape =  (70,) = (M_pca)
    # best_pca_eigenvector.shape =  (70, 2576) = (M_pca,D)
    W_pca = np.zeros(shape=(M_pca, best_pca_eigenvector.shape[1]))
    # W_pca.shape = (M_pca, D)
    for i in range(M_pca):
        W_pca[i] = best_pca_eigenvector[i].copy()
    return mean_face, S,  W_pca

def low_dimension_PCA(X_data, labels):
    num_images = X_data.shape[0]
    D = X_data.shape[1]
    sum_face = np.zeros(shape=X_data.shape[1])
    for i in range(num_images):
        sum_face += X_data[i]
    mean_face = sum_face/num_images

    S_PCA = np.zeros((D, D))
    for i in range(num_images):  # (x-mc).dot((x-mc).T)
        phi_xn = X_data[i] - mean_face
        xn = phi_xn.reshape(D, 1)
        m = xn.dot(xn.T)
        # m.shape = (2576, 2576)
        S_PCA += m
    S_PCA = S_PCA / num_images

    A= np.zeros(shape = (D, num_images))
    for i in range(num_images):
        phi_n = X_data[i] - mean_face
        A[:,i] = phi_n.copy()
    # A.shape =  (2576, 416)
    S = (A.T).dot(A)
    S = S/num_images

    # S.shape =  (416, 416)
    t0 = time.time()
    w, v = np.linalg.eig(S)
    t1 = time.time()
    # w.shape =  (416,
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
    return low_D_dimension_eigenvectors, S_PCA, mean_face, decrease_order_pair_eigenvalue_eigenvector

def low_dimension_PCA_matrix(data, label, M_pca):
    low_D_dimension_eigenvectors, S_PCA, mean_face, decrease_order_pair_eigenvalue_eigenvector = low_dimension_PCA(data, label)
    # decrease_order_pair_eigenvalue_eigenvector[0][1].shape = (104,)

    best_pca_eigenvector = np.array(low_D_dimension_eigenvectors[:M_pca])
    W_pca = np.zeros(shape=(M_pca, best_pca_eigenvector.shape[1]))
    # W_pca.shape = (M_pca, D)
    for i in range(M_pca):
        W_pca[i] = best_pca_eigenvector[i].copy()
    return mean_face, S_PCA, W_pca

def merge_2_PCA_models(meanface1,num_img1, s1, P1_transpose, meanface2, num_img2,s2, P2_transpose, M_pca):

    num_img3 = num_img1 + num_img2
    meanface3 = (num_img1*meanface1 + num_img2*meanface2)/num_img3
    tmp = (meanface1 - meanface2).reshape((meanface1.shape[0],1))
    tmp2 = tmp.dot(tmp.T)
    # tmp2.shape =  (2576, 2576) = (D,D)
    s3 = (num_img1/num_img3)*s1 + (num_img2/num_img3)*s2 + (num_img1*num_img2/(num_img3*num_img3))*tmp2
    # s3.shape = (2576,2576) = (D,D)

    P3_tam = np.zeros(shape = ((P1_transpose.shape[0] + P2_transpose.shape[0] + 1), P1_transpose.shape[1]))
    for i in range(P1_transpose.shape[0]):
        P3_tam[i] = P1_transpose[i].copy()
    for i in range(P2_transpose.shape[0]):
        P3_tam[P1_transpose.shape[0]+i] = P2_transpose[i].copy()
    P3_tam[P1_transpose.shape[0] + P2_transpose.shape[0]] = meanface1 - meanface2
    P3_tmp2 = P3_tam.T
    # P3_tam2.shape =  (2576,141) = (D, d1+d2+1)

    phi , r0 = np.linalg.qr(P3_tmp2)  # choose between P3_tmp2 and P3_tmp2

    # when we want to drop some eigenvectors from phi, then we just do something like this: phi = phi[:,0:100], that mean we keep 100 most important eigenvectors and reject the others
    # phi.shape =  (2576, 141) = (D, d1+d2+1)
    # r0.shape = (141, 141) = (d1+d2+1, d1+d2+1)
    # phi[:,0].dot(phi[:,1].T) =  -2.42861286636753e-17 => orthogonal


    phi_s3_phi = (phi.T).dot(s3).dot(phi)
    # phi_s3_phi.shape =  (141, 141)

    t2 = time.time()
    w, v = np.linalg.eig(phi_s3_phi)
    t3 = time.time()
    # w.shape = (141,)
    # v.shape =  (141, 141)
    pair_phi_s3_eigenvalue_eigenvector = []
    for i in range(w.shape[0]):
        pair_phi_s3_eigenvalue_eigenvector.append((w[i], v[:, i]))  # be careful
    decrease_order_pair_phi_s3_eigenvalue_eigenvector= sorted(pair_phi_s3_eigenvalue_eigenvector, key=lambda x: x[0], reverse=True)
    best_phi_s3_phi_eigenvalues = np.array([ob[0] for ob in decrease_order_pair_phi_s3_eigenvalue_eigenvector])

    decrease_eigenvalues = [abs(ob[0]) for ob in decrease_order_pair_phi_s3_eigenvalue_eigenvector]
    n = 0  # n is the number of non-zero eigenvalues corresponding to eigenvectors in phi
    for i in range(len(decrease_eigenvalues)):
        if decrease_eigenvalues[i] > 0.000001:
            n+= 1

    best_phi_s3_phi_eigenvectors = np.array([ob[1] for ob in decrease_order_pair_phi_s3_eigenvalue_eigenvector])
    # best_phi_s3_eigenvectors.shape =  (141, 141)
    # best_phi_s3_eigenvectors[0] =  [ 6.94126068e-01 -6.72446007e-01 -9.72133073e-03  5.40287186e-02 ... -3.47671725e-02]
    R_tmp = np.zeros(shape = (n, phi.shape[1]))
    for i in range(R_tmp.shape[0]):
        R_tmp[i] = best_phi_s3_phi_eigenvectors[i].copy()
    R = R_tmp.T # columns of R are eigenvectors of phi_S3_phi
    # R.shape = (141,n) = (d1+d2+1, n)
    # R[:,0] =  [ 6.94126068e-01 -6.72446007e-01 -9.72133073e-03  5.40287186e-02 ... -3.47671725e-02]
    P3 = phi.dot(R)
    W_pca_merge_tmp = P3.T  # be careful, W_pca that we need to return is the transpose of P3
    # W_pca_merge_tmp.shape =  (n, 2576) , the form is (num_eigenvectors, D)
    W_pca_merge = W_pca_merge_tmp[:M_pca,:]
    # W_pca_merge.shape = (min(n,M_pca), 2576)
    return meanface3, num_img3, s3, W_pca_merge


def PCA_transform(data, W_pca_matrix, mean_face_vector): # for example, data.shape = (num_img, D), W_pca_matrix.shape = (M_pca,D)
    data_copy = data.copy()
    target_data = np.zeros(shape = (data.shape[0], W_pca_matrix.shape[0])) # shape is (num_img, M_pca)
    for i in range(data_copy.shape[0]):
        tmp = data_copy[i] - mean_face_vector
        # tmp.shape = (D,)
        tmp2 = W_pca_matrix.dot(tmp.T)
        # tmp2.shape = (M_pca,)
        target_data[i] = tmp2.copy()
    return target_data # shape is (num_img, M_pca)


def predict_set(train_data_X, train_data_l, W_pca, mean_face_, test_data_X, test_data_l):
    pca_train = PCA_transform(train_data_X,W_pca, mean_face_)
    pca_test = PCA_transform(test_data_X, W_pca, mean_face_)
    l_preds = []
    for i in range(pca_test.shape[0]):
        x_i = pca_test[i].copy()
        list_distance = []
        for j in range(pca_train.shape[0]):
            x_j = pca_train[j].copy()
            dist_i_j = np.linalg.norm(x_i - x_j)
            list_distance.append((train_data_l[j], dist_i_j))
        increasing_list_distance = sorted(list_distance, key= lambda x: x[1])
        l_preds.append(increasing_list_distance[0][0])

    l_preds = np.array(l_preds)
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(l_preds, test_data_l)
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
    return acc*100

def incremental_PCA(M_pca):
    t0_incremental_PCA = time.time()
    num_components_pca = M_pca
    mean_face_sub1, S1, W_pca_sub1 = low_dimension_PCA_matrix(sub_trainX_1, sub_train_l_1, num_components_pca)
    mean_face_sub2, S2, W_pca_sub2 = low_dimension_PCA_matrix(sub_trainX_2, sub_train_l_2, num_components_pca)
    mean_face_sub3, S3, W_pca_sub3 = low_dimension_PCA_matrix(sub_trainX_3, sub_train_l_3, num_components_pca)
    mean_face_sub4, S4, W_pca_sub4 = low_dimension_PCA_matrix(sub_trainX_4, sub_train_l_4, num_components_pca)
    # mean_face_sub1.shape =  (2576,)
    # mean_face_sub1.shape[0] =  2576
    # W_pca.shape = (M_pca, D)
    a0 = time.time()
    mean_face_merge12, num_img_merge12, s_merge12, w_pca_merge12 = merge_2_PCA_models(mean_face_sub1, N1, S1, W_pca_sub1, mean_face_sub2, N2, S2, W_pca_sub2, M_pca)
    mean_face_merge123, num_img_merge123, s_merge123, w_pca_merge123 = merge_2_PCA_models(mean_face_merge12, num_img_merge12, s_merge12, w_pca_merge12, mean_face_sub3, N3, S3, W_pca_sub3, M_pca)
    mean_face_merge1234, num_img_merge1234, s_merge1234, w_pca_merge1234 = merge_2_PCA_models(mean_face_merge123, num_img_merge123, s_merge123, w_pca_merge123, mean_face_sub4, N4, S4, W_pca_sub4, M_pca)
    # w_pca_merge1234.shape =  (num_eigenvectors, 2576)
    a1 = time.time()

    t1_incremental_PCA = time.time()
    acc = predict_set(train_X, train_l, w_pca_merge1234,mean_face_merge1234, test_X, test_l)

    return mean_face_merge1234, w_pca_merge1234
#mean_face_merge1234, w_pca_merge1234 = incremental_PCA(M_pca= 150)

def batch_PCA(M_pca):
    mean_face, S, W_pca = low_dimension_PCA_matrix(train_X, train_l, M_pca)
    acc= predict_set(train_X, train_l, W_pca, mean_face, test_X, test_l)
    return mean_face, W_pca
#batch_PCA(M_pca = 150)

def PCA_only_first_subset(M_pca):
    mean_face_sub1, S1, W_pca_sub1 = low_dimension_PCA_matrix(sub_trainX_1, sub_train_l_1, M_pca)
    acc = predict_set(train_X, train_l, W_pca_sub1, mean_face_sub1, test_X, test_l)
    return mean_face_sub1, W_pca_sub1

#PCA_only_first_subset(M_pca = 70)

def reconstruct_single_img(img, mean_face_, w_pca, h = 10, w = 10):
    # w_pca.shape = (M_pca, D)
    copy_img = img.copy()
    xn = copy_img.flatten()

    num_pca_base = w_pca.shape[0]
    list_eigenvectors = []
    for i in range(w_pca.shape[0]):
        ui = w_pca[i].copy()
        list_eigenvectors.append(ui)
    phi_xn = xn - mean_face_

    tmp_reconstructed_face_i = mean_face_.copy()
    for i in range(len(list_eigenvectors)):
        a_n_i = (phi_xn.T).dot(list_eigenvectors[i])
        tmp = a_n_i*list_eigenvectors[i]
        tmp_reconstructed_face_i += tmp
    reconstructed_face_i = tmp_reconstructed_face_i.reshape((h, w))
    #plt.imshow(reconstructed_face_i.T)
    #plt.show()
    return tmp_reconstructed_face_i # shape is (D,)


def compute_mean_reconstruct_error(mean_face_, w_pca_, test_set): # test_set.shape =  (416, 2576) = (n, D)
    list_reconstruct_error_for_test_set = []
    for i in range(test_set.shape[0]):
        img_ = test_set[i].copy()
        reconstruct_ = reconstruct_single_img(img_, mean_face_, w_pca_)
        reconstruct_error = (np.linalg.norm(img_ - reconstruct_))**2
        list_reconstruct_error_for_test_set.append(reconstruct_error)
    list_reconstruct_error_for_test_set = np.array(list_reconstruct_error_for_test_set)
    mean_reconstruct_error_for_test_set = np.mean(list_reconstruct_error_for_test_set)
    return mean_reconstruct_error_for_test_set

def reconstruction_error_for_3_model(M_pca):
    """
    mean_face_merge1234, w_pca_merge1234 = incremental_PCA(M_pca)
    incremental_pca_error = compute_mean_reconstruct_error(mean_face_merge1234, w_pca_merge1234, test_X)

    mean_face_batch, w_pca_batch = batch_PCA(M_pca)
    batch_error = compute_mean_reconstruct_error(mean_face_batch, w_pca_batch, test_X)"""

    mean_face_sub1, w_pca_sub1 = PCA_only_first_subset(M_pca)
    only_first_error = compute_mean_reconstruct_error(mean_face_sub1, w_pca_sub1, test_X)

t_start = time.time()
reconstruction_error_for_3_model(50)
t_end = time.time()
print("execution time: ", t_end - t_start)










































































