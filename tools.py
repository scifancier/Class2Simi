import numpy as np
import utils
import torch


def class2simi(transition_matrix):
    v00 = v01 = v10 = v11 = 0
    t = transition_matrix
    num_classes = transition_matrix.shape[0]
    for i in range(num_classes):
        for j in range(num_classes):
            a = t[i][j]
            for m in range(num_classes):
                for n in range(num_classes):
                    b = t[m][n]
                    if i == m and j == n:
                        v11 += a * b
                    if i == m and j != n:
                        v10 += a * b
                    if i != m and j == n:
                        v01 += a * b
                    if i != m and j != n:
                        v00 += a * b
    simi_T = np.zeros([2, 2])
    simi_T[0][0] = v11 / (v11 + v10)
    simi_T[0][1] = v10 / (v11 + v10)
    simi_T[1][0] = v01 / (v01 + v00)
    simi_T[1][1] = v00 / (v01 + v00)
    # print(simi_T)
    return simi_T


def norm(T):
    row_sum = np.sum(T, 1)
    T_norm = T / row_sum
    return T_norm


def error(T, T_true):
    error = np.sum(np.abs(T - T_true)) / np.sum(np.abs(T_true))
    return error


def rand_transition_matrix_generate(noise_rate=0.5, num_classes=10):
    np.random.seed(1)
    t = np.random.rand(num_classes, num_classes)
    i = np.eye(num_classes)
    if noise_rate == 0.1:
        t = t + 3.0 * num_classes * i
    if noise_rate == 0.3:
        t = t + 1.2 * num_classes * i
    for a in range(num_classes):
        t[a] = t[a] / t[a].sum()

    P = np.asarray(t)

    return P


def pair_transition_matrix_generate(noise_rate=0.5, num_classes=10):
    P = np.eye(num_classes)
    n = noise_rate

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, num_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[num_classes - 1, num_classes - 1], P[num_classes - 1, 0] = 1. - n, n
    return P


def s_transition_matrix_generate(noise_rate=0.5, num_classes=10):
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, num_classes - 1):
            P[i, i] = 1. - n
        P[num_classes - 1, num_classes - 1] = 1. - n
    return P


def fit(X, num_classes, per_radio=97):
    # number of classes
    c = num_classes
    T = np.empty((c, c))
    eta_corr = X
    for i in np.arange(c):
        eta_thresh = np.percentile(eta_corr[:, i], per_radio, interpolation='higher')
        robust_eta = eta_corr[:, i]
        robust_eta[robust_eta >= eta_thresh] = 0.0
        idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            T[i, j] = eta_corr[idx_best, j]
    return T


# flip clean labels to noisy labels
# train set and val set split
def dataset_split(train_images, train_labels, noise_rate=0.5, split_per=0.9, random_seed=1, noise_type='none',
                  num_classes=10):
    clean_train_labels = train_labels[:, np.newaxis]
    if noise_type == 'none':
        # print('Clean data')
        noisy_labels = clean_train_labels.squeeze()
    if noise_type == 's':
        # print('Symmetric noise')
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_multiclass_symmetric(clean_train_labels,
                                                                                              noise=noise_rate,
                                                                                              random_state=random_seed,
                                                                                              num_classes=num_classes)
        noisy_labels = noisy_labels.squeeze()
    if noise_type == 'as':
        # print('Asymmetric noise')
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_rand(clean_train_labels,
                                                                              noise=noise_rate,
                                                                              random_state=random_seed,
                                                                              num_classes=num_classes)
        noisy_labels = noisy_labels.squeeze()
    if noise_type == 'p':
        print('Pair noise')
        noisy_labels, real_noise_rate, transition_matrix = utils.noisify_pairflip(clean_train_labels,
                                                                                  noise=noise_rate,
                                                                                  random_state=random_seed,
                                                                                  num_classes=num_classes)
        noisy_labels = noisy_labels.squeeze()
    # print(noisy_labels)
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index, :], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels


