import numpy as np
import tensorflow as tf
from .muse_architecture import structured_embedding
from scipy.spatial.distance import pdist
import phenograph

""" Model fitting and feature prediction of MUSE """


def muse_fit_predict(data_x,
                     data_y,
                     label_x,
                     label_y,
                     latent_dim=100,
                     n_epochs=500,
                     lambda_regul=5,
                     lambda_super=5):
    """
        MUSE model fitting and predicting:
          This function is used to train the MUSE model on multi-modality data

        Parameters:
          data_x:       input for transcript modality; matrix of  n * p, where n = number of cells, p = number of genes.
          data_y:       input for morphological modality; matrix of n * q, where n = number of cells, q is the feature dimension.
          label_x:      initial reference cluster label for transcriptional modality.
          label_y:      inital reference cluster label for morphological modality.
          latent_dim:   feature dimension of joint latent representation.
          n_epochs:     maximal epoch used in training.
          lambda_regul: weight for regularization term in the loss function.
          lambda_super: weight for supervised learning loss in the loss function.

        Output:
          latent:       joint latent representation learned by MUSE.
          reconstruct_x:reconstructed feature matrix corresponding to input data_x.
          reconstruct_y:reconstructed feature matrix corresponding to input data_y.
          latent_x:     modality-specific latent representation corresponding to data_x.
          latent_y:     modality-specific latent representation corresponding to data_y.

        Altschuler & Wu Lab 2020.
        Software provided as is under MIT License.
    """

    """ initial parameter setting """
    # parameter setting for neural network
    n_hidden = 128  # number of hidden node in neural network
    learn_rate = 1e-4  # learning rate in the optimization
    batch_size = 64  # number of cells in the training batch
    n_epochs_init = 200  # number of training epoch in model initialization
    print_epochs = 50  # epoch interval to display the current training loss
    cluster_update_epoch = 200  # epoch interval to update modality-specific clusters

    # read data-specific parameters from inputs
    feature_dim_x = data_x.shape[1]
    feature_dim_y = data_y.shape[1]
    n_sample = data_x.shape[0]

    # GPU configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    """ construct computation graph using TensorFlow """
    tf.reset_default_graph()

    # raw data from two modalities
    x = tf.placeholder(tf.float32, shape=[None, feature_dim_x], name='input_x')
    y = tf.placeholder(tf.float32, shape=[None, feature_dim_y], name='input_y')

    # labels inputted for references
    ref_label_x = tf.placeholder(tf.float32, shape=[None], name='ref_label_x')
    ref_label_y = tf.placeholder(tf.float32, shape=[None], name='ref_label_y')

    # hyperparameter in triplet loss
    triplet_lambda = tf.placeholder(tf.float32, name='triplet_lambda')
    triplet_margin = tf.placeholder(tf.float32, name='triplet_margin')

    # network architecture
    z, x_hat, y_hat, encode_x, encode_y, loss, \
    reconstruction_error, weight_penalty, \
    trip_loss_x, trip_loss_y = structured_embedding(x,
                                                    y,
                                                    ref_label_x,
                                                    ref_label_y,
                                                    latent_dim,
                                                    triplet_margin,
                                                    n_hidden,
                                                    lambda_regul,
                                                    triplet_lambda)
    # optimization operator
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    """ MUSE optimization """
    total_batch = int(n_sample / batch_size)
    with tf.device('/gpu:0'):
        with tf.Session(config=config) as sess:

            """ initialization of autoencoder architecture for MUSE """
            print('MUSE initialization')
            # global parameter initialization
            sess.run(tf.global_variables_initializer(), feed_dict={triplet_lambda: 0,
                                                                   triplet_margin: 0})

            for epoch in range(n_epochs_init):
                # randomly permute samples
                random_idx = np.random.permutation(n_sample)
                data_train_x = data_x[random_idx, :]
                data_train_y = data_y[random_idx, :]

                for i in range(total_batch):
                    # input data batches
                    offset = (i * batch_size) % (n_sample)
                    batch_x_input = data_train_x[offset:(offset + batch_size), :]
                    batch_y_input = data_train_y[offset:(offset + batch_size), :]

                    # initialize parameters without self-supervised loss (triplet_lambda=0)
                    sess.run(train_op,
                             feed_dict={x: batch_x_input,
                                        y: batch_y_input,
                                        ref_label_x: np.zeros(batch_x_input.shape[0]),
                                        ref_label_y: np.zeros(batch_y_input.shape[0]),
                                        triplet_lambda: 0,
                                        triplet_margin: 0})

                # calculate and print loss terms for current epoch
                if epoch % print_epochs == 0:
                    L_total, L_reconstruction, L_weight = \
                        sess.run((loss, reconstruction_error, weight_penalty),
                                 feed_dict={x: data_train_x,
                                            y: data_train_y,
                                            ref_label_x: np.zeros(data_train_x.shape[0]),  # no use as triplet_lambda=0
                                            ref_label_y: np.zeros(data_train_y.shape[0]),  # no use as triplet_lambda=0
                                            triplet_lambda: 0,
                                            triplet_margin: 0})

                    print(
                        "epoch: %d, \t total loss: %03.5f,\t reconstruction loss: %03.5f,\t sparse penalty: %03.5f"
                        % (epoch, L_total, L_reconstruction, L_weight))

            # estimate the margin for the triplet loss
            latent, reconstruct_x, reconstruct_y = \
                sess.run((z, x_hat, y_hat),
                         feed_dict={x: data_x,
                                    y: data_y,
                                    ref_label_x: np.zeros(data_x.shape[0]),
                                    ref_label_y: np.zeros(data_y.shape[0]),
                                    triplet_lambda: 0,
                                    triplet_margin: 0})
            latent_pd_matrix = pdist(latent, 'euclidean')
            latent_pd_sort = np.sort(latent_pd_matrix)
            select_top_n = np.int(latent_pd_sort.size * 0.2)
            margin_estimate = np.median(latent_pd_sort[-select_top_n:]) - np.median(latent_pd_sort[:select_top_n])

            # refine MUSE parameters with reference labels and triplet losses
            for epoch in range(n_epochs_init):
                # randomly permute samples
                random_idx = np.random.permutation(n_sample)
                data_train_x = data_x[random_idx, :]
                data_train_y = data_y[random_idx, :]
                label_train_x = label_x[random_idx]
                label_train_y = label_y[random_idx]

                for i in range(total_batch):
                    # data batches
                    offset = (i * batch_size) % (n_sample)
                    batch_x_input = data_train_x[offset:(offset + batch_size), :]
                    batch_y_input = data_train_y[offset:(offset + batch_size), :]
                    label_x_input = label_train_x[offset:(offset + batch_size)]
                    label_y_input = label_train_y[offset:(offset + batch_size)]

                    # refine parameters
                    sess.run(train_op,
                             feed_dict={x: batch_x_input,
                                        y: batch_y_input,
                                        ref_label_x: label_x_input,
                                        ref_label_y: label_y_input,
                                        triplet_lambda: lambda_super,
                                        triplet_margin: margin_estimate})

                # calculate loss on all input data for current epoch
                if epoch % print_epochs == 0:
                    L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y = \
                        sess.run((loss, reconstruction_error, weight_penalty, trip_loss_x, trip_loss_y),
                                 feed_dict={x: data_train_x,
                                            y: data_train_y,
                                            ref_label_x: label_train_x,
                                            ref_label_y: label_train_y,
                                            triplet_lambda: lambda_super,
                                            triplet_margin: margin_estimate})

                    print(
                        "epoch: %d, \t total loss: %03.5f,\t reconstruction loss: %03.5f,\t sparse penalty: %03.5f,\t x triplet: %03.5f,\t y triplet: %03.5f"
                        % (epoch, L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y))

            # update cluster labels based modality-specific latents
            latent_x, latent_y = \
                sess.run((encode_x, encode_y),
                         feed_dict={x: data_x,
                                    y: data_y,
                                    ref_label_x: label_x,
                                    ref_label_y: label_y,
                                    triplet_lambda: lambda_super,
                                    triplet_margin: margin_estimate})

            # update cluster labels using PhenoGraph
            label_x_update, _, _ = phenograph.cluster(latent_x)
            label_y_update, _, _ = phenograph.cluster(latent_y)
            print('Finish initialization of MUSE')

            ''' Training of MUSE '''
            for epoch in range(n_epochs):
                # randomly permute samples
                random_idx = np.random.permutation(n_sample)
                data_train_x = data_x[random_idx, :]
                data_train_y = data_y[random_idx, :]
                label_train_x = label_x_update[random_idx]
                label_train_y = label_y_update[random_idx]

                # loop over all batches
                for i in range(total_batch):
                    # batch data
                    offset = (i * batch_size) % (n_sample)
                    batch_x_input = data_train_x[offset:(offset + batch_size), :]
                    batch_y_input = data_train_y[offset:(offset + batch_size), :]
                    batch_label_x_input = label_train_x[offset:(offset + batch_size)]
                    batch_label_y_input = label_train_y[offset:(offset + batch_size)]

                    sess.run(train_op,
                             feed_dict={x: batch_x_input,
                                        y: batch_y_input,
                                        ref_label_x: batch_label_x_input,
                                        ref_label_y: batch_label_y_input,
                                        triplet_lambda: lambda_super,
                                        triplet_margin: margin_estimate})

                # calculate and print losses on whole training dataset
                if epoch % print_epochs == 0:
                    L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y = \
                        sess.run((loss, reconstruction_error, weight_penalty, trip_loss_x, trip_loss_y),
                                 feed_dict={x: data_train_x,
                                            y: data_train_y,
                                            ref_label_x: label_train_x,
                                            ref_label_y: label_train_y,
                                            triplet_lambda: lambda_super,
                                            triplet_margin: margin_estimate})
                    # print cost every epoch
                    print(
                        "epoch: %d, \t total loss: %03.5f,\t reconstruction loss: %03.5f,\t sparse penalty: %03.5f,\t x triplet loss: %03.5f,\t y triplet loss: %03.5f"
                        % (epoch, L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y))

                # update cluster labels based on new modality-specific latent representations
                if epoch % cluster_update_epoch == 0:
                    latent_x, latent_y = \
                        sess.run((encode_x, encode_y),
                                 feed_dict={x: data_x,
                                            y: data_y,
                                            ref_label_x: label_x,
                                            ref_label_y: label_y,
                                            triplet_lambda: lambda_super,
                                            triplet_margin: margin_estimate})

                    # use PhenoGraph to obtain cluster label
                    label_x_update, _, _ = phenograph.cluster(latent_x)
                    label_y_update, _, _ = phenograph.cluster(latent_y)

            """ MUSE output """
            latent, reconstruct_x, reconstruct_y, latent_x, latent_y = \
                sess.run((z, x_hat, y_hat, encode_x, encode_y),
                         feed_dict={x: data_x,
                                    y: data_y,
                                    ref_label_x: label_x,  # no effects to representations
                                    ref_label_y: label_y,  # no effects to representations
                                    triplet_lambda: lambda_super,
                                    triplet_margin: margin_estimate})

    return latent, reconstruct_x, reconstruct_y, latent_x, latent_y
