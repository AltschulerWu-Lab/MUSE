from __future__ import print_function
import numpy as np
import random
from copy import deepcopy

# for the replication of the same simulation data, uncomment next line
# np.random.seed(2020)


""" Function to generate simulation data with two modalities """


def multi_modal_simulator(n_clusters,
                          n,
                          d_1,
                          d_2,
                          k,
                          sigma_1,
                          sigma_2,
                          decay_coef_1,
                          decay_coef_2,
                          merge_prob
                          ):
    """
    Generate simulated data with two modalities.

    Parameters:
      n_clusters:       number of ground truth clusters.
      n:                number of cells to simulate.
      d_1:              dimension of features for transcript modality.
      d_2:              dimension of features for morphological modality.
      k:                dimension of latent code to generate simulate data (for both modality)
      sigma_1:          variance of gaussian noise for transcript modality.
      sigma_2:          variance of gaussian noise for morphological modality.
      decay_coef_1:     decay coefficient of dropout rate for transcript modality.
      decay_coef_2:     decay coefficient of dropout rate for morphological modality.
      merge_prob:       probability to merge neighbor clusters for the generation of modality-specific
                        clusters (same for both modalities)


    Output:
      a dataframe with keys as follows

      'true_cluster':   true cell clusters, a vector of length n

      'data_a_full':    feature matrix of transcript without dropouts
      'data_a_dropout': feature matrix of transcript with dropouts
      'data_a_label':   cluster labels to generate transcript features after merging

      'data_b_full':    feature matrix of morphology without dropouts
      'data_b_dropout': feature matrix of morphology with dropouts
      'data_b_label':   cluster labels to generate morphological features after merging


    Feng Bao @ Altschuler & Wu Lab @ UCSF 2022..
    Software provided as is under MIT License.
    """

    # data dict for output
    data = {}

    """ generation of true cluster labels """
    cluster_ids = np.array([random.choice(range(n_clusters)) for i in range(n)])
    data['true_cluster'] = cluster_ids

    """ merge a group of true clusters randomly """
    # divide clusters into two equal groups
    section_a = np.arange(np.floor(n_clusters / 2.0))
    section_b = np.arange(np.floor(n_clusters / 2.0), n_clusters)

    uniform_a = np.random.uniform(size=section_a.size - 1)
    uniform_b = np.random.uniform(size=section_b.size - 1)

    section_a_cp = section_a.copy()
    section_b_cp = section_b.copy()

    # randomly merge two neighbor clusters at merge_prob
    for i in range(uniform_a.size):
        if uniform_a[i] < merge_prob:
            section_a_cp[i + 1] = section_a_cp[i]
    for i in range(uniform_b.size):
        if uniform_b[i] < merge_prob:
            section_b_cp[i + 1] = section_b_cp[i]
    # reindex
    cluster_ids_a = cluster_ids.copy()
    cluster_ids_b = cluster_ids.copy()
    for i in range(section_a.size):
        idx = np.nonzero(cluster_ids == section_a[i])[0]
        cluster_ids_a[idx] = section_a_cp[i]
    for i in range(section_b.size):
        idx = np.nonzero(cluster_ids == section_b[i])[0]
        cluster_ids_b[idx] = section_b_cp[i]

    """ Simulation of transcriptional modality """
    # generate latent code
    Z_a = np.zeros([k, n])
    for id in list(set(cluster_ids_a)):
        idxs = cluster_ids_a == id
        cluster_mu = np.random.random([k]) - 0.5
        Z_a[:, idxs] = np.random.multivariate_normal(mean=cluster_mu, cov=0.1 * np.eye(k),
                                                     size=idxs.sum()).transpose()
    # random projection matrix
    A_a = np.random.random([d_1, k]) - 0.5
    # gaussian noise
    noise_a = np.random.normal(0, sigma_1, size=[d_1, n])
    # raw feature
    X_a = np.dot(A_a, Z_a).transpose()
    X_a[X_a < 0] = 0
    # dropout
    cutoff = np.exp(-decay_coef_1 * (X_a ** 2))
    X_a = X_a + noise_a.T
    X_a[X_a < 0] = 0
    Y_a = deepcopy(X_a)
    rand_matrix = np.random.random(Y_a.shape)
    zero_mask = rand_matrix < cutoff
    Y_a[zero_mask] = 0

    data['data_a_full'] = X_a
    data['data_a_dropout'] = Y_a
    data['data_a_label'] = cluster_ids_a

    """ Simulation of morphological modality """
    # generate latent code
    Z_b = np.zeros([k, n])
    for id in list(set(cluster_ids_b)):
        idxs = cluster_ids_b == id
        cluster_mu = (np.random.random([k]) - .5)
        Z_b[:, idxs] = np.random.multivariate_normal(mean=cluster_mu, cov=0.1 * np.eye(k),
                                                     size=idxs.sum()).transpose()

    # first layer of neural network
    A_b_1 = np.random.random([d_2, k]) - 0.5
    X_b_1 = np.dot(A_b_1, Z_b)
    X_b_1 = 1 / (1 + np.exp(-X_b_1))

    # second layer of neural network
    A_b_2 = np.random.random([d_2, d_2]) - 0.5
    noise_b = np.random.normal(0, sigma_2, size=[d_2, n])
    X_b = (np.dot(A_b_2, X_b_1) + noise_b).transpose()
    X_b = 1 / (1 + np.exp(-X_b))

    # random dropouts
    Y_b = deepcopy(X_b)
    rand_matrix = np.random.random(Y_b.shape)
    zero_mask = rand_matrix < decay_coef_2
    Y_b[zero_mask] = 0

    data['data_b_full'] = X_b
    data['data_b_dropout'] = Y_b
    data['data_b_label'] = cluster_ids_b

    return data
    
def multi_modal_simulator_3_modalities(n_clusters,
                                       n,
                                       d_1,
                                       d_2,
                                       d_3,
                                       k,
                                       sigma_1,
                                       sigma_2,
                                       sigma_3,
                                       decay_coef_1,
                                       decay_coef_2,
                                       decay_coef_3,
                                       merge_prob
                                       ):
    """
    Generate simulated data with two modalities.
    Parameters:
      n_clusters:       number of ground truth clusters.
      n:                number of cells to simulate.
      d_1:              dimension of features for transcript modality.
      d_2:              dimension of features for morphological modality.
      d_3:              dimension of features for 3rd modality.
      k:                dimension of latent code to generate simulate data (for both modality)
      sigma_1:          variance of gaussian noise for transcript modality.
      sigma_2:          variance of gaussian noise for morphological modality.
      sigma_3:          variance of gaussian noise for 3rd modality.
      decay_coef_1:     decay coefficient of dropout rate for transcript modality.
      decay_coef_2:     decay coefficient of dropout rate for morphological modality.
      decay_coef_3:     decay coefficient of dropout rate for 3rd modality.
      merge_prob:       probability to merge neighbor clusters for the generation of modality-specific
                        clusters (same for both modalities)
    Output:
      a dataframe with keys as follows
      'true_cluster':   true cell clusters, a vector of length n
      'data_a_full':    feature matrix of transcript without dropouts
      'data_a_dropout': feature matrix of transcript with dropouts
      'data_a_label':   cluster labels to generate transcript features after merging
      'data_b_full':    feature matrix of morphology without dropouts
      'data_b_dropout': feature matrix of morphology with dropouts
      'data_b_label':   cluster labels to generate morphological features after merging
      'data_c_full':    feature matrix of 3rd modality without dropouts
      'data_c_dropout': feature matrix of 3rd modality with dropouts
      'data_c_label':   cluster labels to generate 3rd modality features after merging
    Feng Bao (fbao0110ATgmail.com) @ Altschuler & Wu Lab 2022.
    Software provided as is under MIT License.
    """

    # data dict for output
    data = {}

    """ generation of true cluster labels """
    cluster_ids = np.array([random.choice(range(n_clusters)) for i in range(n)])
    data['true_cluster'] = cluster_ids

    """ merge a group of true clusters randomly """
    # divide clusters into two equal groups
    section_a = np.arange(np.floor(n_clusters / 3.0))
    section_b = np.arange(np.floor(n_clusters / 3.0), np.floor(n_clusters / 3.0 * 2))
    section_c = np.arange(np.floor(n_clusters / 3.0 * 2), n_clusters)

    uniform_a = np.random.uniform(size=section_a.size - 1)
    uniform_b = np.random.uniform(size=section_b.size - 1)
    uniform_c = np.random.uniform(size=section_c.size - 1)

    section_a_cp = section_a.copy()
    section_b_cp = section_b.copy()
    section_c_cp = section_c.copy()

    # randomly merge two neighbor clusters at merge_prob
    for i in range(uniform_a.size):
        if uniform_a[i] < merge_prob:
            section_a_cp[i + 1] = section_a_cp[i]
    for i in range(uniform_b.size):
        if uniform_b[i] < merge_prob:
            section_b_cp[i + 1] = section_b_cp[i]
    for i in range(uniform_c.size):
        if uniform_c[i] < merge_prob:
            section_c_cp[i + 1] = section_c_cp[i]
    # reindex
    cluster_ids_a = cluster_ids.copy()
    cluster_ids_b = cluster_ids.copy()
    cluster_ids_c = cluster_ids.copy()
    for i in range(section_a.size):
        idx = np.nonzero(cluster_ids == section_a[i])[0]
        cluster_ids_a[idx] = section_a_cp[i]
    for i in range(section_b.size):
        idx = np.nonzero(cluster_ids == section_b[i])[0]
        cluster_ids_b[idx] = section_b_cp[i]
    for i in range(section_c.size):
        idx = np.nonzero(cluster_ids == section_c[i])[0]
        cluster_ids_c[idx] = section_c_cp[i]

    """ Simulation of transcriptional modality """
    # generate latent code
    Z_a = np.zeros([k, n])
    for id in list(set(cluster_ids_a)):
        idxs = cluster_ids_a == id
        cluster_mu = np.random.random([k]) - 0.5
        Z_a[:, idxs] = np.random.multivariate_normal(mean=cluster_mu, cov=0.1 * np.eye(k),
                                                     size=idxs.sum()).transpose()
    # random projection matrix
    A_a = np.random.random([d_1, k]) - 0.5
    # gaussian noise
    noise_a = np.random.normal(0, sigma_1, size=[d_1, n])
    # raw feature
    X_a = np.dot(A_a, Z_a).transpose()
    X_a[X_a < 0] = 0
    # dropout
    cutoff = np.exp(-decay_coef_1 * (X_a ** 2))
    X_a = X_a + noise_a.T
    X_a[X_a < 0] = 0
    Y_a = deepcopy(X_a)
    rand_matrix = np.random.random(Y_a.shape)
    zero_mask = rand_matrix < cutoff
    Y_a[zero_mask] = 0

    data['data_a_full'] = X_a
    data['data_a_dropout'] = Y_a
    data['data_a_label'] = cluster_ids_a

    """ Simulation of morphological modality """
    # generate latent code
    Z_b = np.zeros([k, n])
    for id in list(set(cluster_ids_b)):
        idxs = cluster_ids_b == id
        cluster_mu = (np.random.random([k]) - .5)
        Z_b[:, idxs] = np.random.multivariate_normal(mean=cluster_mu, cov=0.1 * np.eye(k),
                                                     size=idxs.sum()).transpose()

    # first layer of neural network
    A_b_1 = np.random.random([d_2, k]) - 0.5
    X_b_1 = np.dot(A_b_1, Z_b)
    X_b_1 = 1 / (1 + np.exp(-X_b_1))

    # second layer of neural network
    A_b_2 = np.random.random([d_2, d_2]) - 0.5
    noise_b = np.random.normal(0, sigma_2, size=[d_2, n])
    X_b = (np.dot(A_b_2, X_b_1) + noise_b).transpose()
    X_b = 1 / (1 + np.exp(-X_b))

    # random dropouts
    Y_b = deepcopy(X_b)
    rand_matrix = np.random.random(Y_b.shape)
    zero_mask = rand_matrix < decay_coef_2
    Y_b[zero_mask] = 0

    data['data_b_full'] = X_b
    data['data_b_dropout'] = Y_b
    data['data_b_label'] = cluster_ids_b

    """ Simulation of the third modality """
    # generate latent code
    Z_c = np.zeros([k, n])
    for id in list(set(cluster_ids_c)):
        idxs = cluster_ids_c == id
        cluster_mu = (np.random.random([k]) - .5)
        Z_c[:, idxs] = np.random.multivariate_normal(mean=cluster_mu, cov=0.1 * np.eye(k),
                                                     size=idxs.sum()).transpose()

    # first layer of neural network
    A_c_1 = np.random.random([d_3, k]) - 0.5
    X_c_1 = np.dot(A_c_1, Z_c)
    X_c_1 = 1 / (1 + np.exp(-X_c_1))

    # second layer of neural network
    A_c_2 = np.random.random([d_3, d_3]) - 0.5
    noise_c = np.random.normal(0, sigma_3, size=[d_3, n])
    X_c = (np.dot(A_c_2, X_c_1) + noise_c).transpose()
    X_c = 1 / (1 + np.exp(-X_c))

    # random dropouts
    Y_c = deepcopy(X_c)
    rand_matrix = np.random.random(Y_c.shape)
    zero_mask = rand_matrix < decay_coef_3
    Y_c[zero_mask] = 0

    data['data_c_full'] = X_c
    data['data_c_dropout'] = Y_c
    data['data_c_label'] = cluster_ids_c

    return data