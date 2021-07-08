import numpy as np
from .muse_architecture import structured_embedding
from scipy.spatial.distance import pdist
import phenograph
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

""" Model fitting and feature prediction of MUSE """


def muse_fit_predict(data_x,
                     data_y,
                     data_z,
                     label_x,
                     label_y,
                     label_z,
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
          data_z:       input for the third modality; matrix of n * r, where n = number of cells, r is the feature dimension.
          label_x:      initial reference cluster label for transcriptional modality.
          label_y:      inital reference cluster label for morphological modality.
          label_z:      inital reference cluster label for the third modality.
          latent_dim:   feature dimension of joint latent representation.
          n_epochs:     maximal epoch used in training.
          lambda_regul: weight for regularization term in the loss function.
          lambda_super: weight for supervised learning loss in the loss function.

        Output:
          latent:       joint latent representation learned by MUSE.
          reconstruct_x:reconstructed feature matrix corresponding to input data_x.
          reconstruct_y:reconstructed feature matrix corresponding to input data_y.
          reconstruct_z:reconstructed feature matrix corresponding to input data_z.
          latent_x:     modality-specific latent representation corresponding to data_x.
          latent_y:     modality-specific latent representation corresponding to data_y.
          latent_z    modality-specific latent representation corresponding to data_z.

        Feng Bao @ Altschuler & Wu Lab 2021.
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
    feature_dim_z = data_z.shape[1]
    n_sample = data_x.shape[0]

    # GPU configuration
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    """ construct computation graph using TensorFlow """
    tf.reset_default_graph()

    # raw data from two modalities
    x = tf.placeholder(tf.float32, shape=[None, feature_dim_x], name='input_x')
    y = tf.placeholder(tf.float32, shape=[None, feature_dim_y], name='input_y')
    z = tf.placeholder(tf.float32, shape=[None, feature_dim_z], name='input_z')

    # labels inputted for references
    ref_label_x = tf.placeholder(tf.float32, shape=[None], name='ref_label_x')
    ref_label_y = tf.placeholder(tf.float32, shape=[None], name='ref_label_y')
    ref_label_z = tf.placeholder(tf.float32, shape=[None], name='ref_label_z')

    # hyperparameter in triplet loss
    triplet_lambda = tf.placeholder(tf.float32, name='triplet_lambda')
    triplet_margin = tf.placeholder(tf.float32, name='triplet_margin')

    # network architecture
    joint_latent, x_hat, y_hat, z_hat, \
    encode_x, encode_y, encode_z, \
    loss, reconstruction_error, weight_penalty, \
    trip_loss_x, trip_loss_y, trip_loss_z = structured_embedding(x,
                                                                 y,
                                                                 z,
                                                                 ref_label_x,
                                                                 ref_label_y,
                                                                 ref_label_z,
                                                                 latent_dim,
                                                                 triplet_margin,
                                                                 n_hidden,
                                                                 lambda_regul,
                                                                 triplet_lambda)
    # optimization operator
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    print('++++++++++ MUSE for multi-modality single-cell analysis ++++++++++')
    """ MUSE optimization """
    total_batch = int(n_sample / batch_size)

    with tf.Session() as sess:

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
            data_train_z = data_z[random_idx, :]

            for i in range(total_batch):
                # input data batches
                offset = (i * batch_size) % (n_sample)
                batch_x_input = data_train_x[offset:(offset + batch_size), :]
                batch_y_input = data_train_y[offset:(offset + batch_size), :]
                batch_z_input = data_train_z[offset:(offset + batch_size), :]

                # initialize parameters without self-supervised loss (triplet_lambda=0)
                sess.run(train_op,
                         feed_dict={x: batch_x_input,
                                    y: batch_y_input,
                                    z: batch_z_input,
                                    ref_label_x: np.zeros(batch_x_input.shape[0]),
                                    ref_label_y: np.zeros(batch_y_input.shape[0]),
                                    ref_label_z: np.zeros(batch_z_input.shape[0]),
                                    triplet_lambda: 0,
                                    triplet_margin: 0})

            # calculate and print loss terms for current epoch
            if epoch % print_epochs == 0:
                L_total, L_reconstruction, L_weight = \
                    sess.run((loss, reconstruction_error, weight_penalty),
                             feed_dict={x: data_train_x,
                                        y: data_train_y,
                                        z: data_train_z,
                                        ref_label_x: np.zeros(data_train_x.shape[0]),  # no use as triplet_lambda=0
                                        ref_label_y: np.zeros(data_train_y.shape[0]),  # no use as triplet_lambda=0
                                        ref_label_z: np.zeros(data_train_z.shape[0]),  # no use as triplet_lambda=0
                                        triplet_lambda: 0,
                                        triplet_margin: 0})

                print(
                    "epoch: %d, \t total loss: %03.5f,\t reconstruction loss: %03.5f,\t sparse penalty: %03.5f"
                    % (epoch, L_total, L_reconstruction, L_weight))

        # estimate the margin for the triplet loss
        latent, reconstruct_x, reconstruct_y, reconstruct_z = \
            sess.run((joint_latent, x_hat, y_hat, z_hat),
                     feed_dict={x: data_x,
                                y: data_y,
                                z: data_z,
                                ref_label_x: np.zeros(data_x.shape[0]),
                                ref_label_y: np.zeros(data_y.shape[0]),
                                ref_label_z: np.zeros(data_z.shape[0]),
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
            data_train_z = data_z[random_idx, :]
            label_train_x = label_x[random_idx]
            label_train_y = label_y[random_idx]
            label_train_z = label_z[random_idx]

            for i in range(total_batch):
                # data batches
                offset = (i * batch_size) % (n_sample)
                batch_x_input = data_train_x[offset:(offset + batch_size), :]
                batch_y_input = data_train_y[offset:(offset + batch_size), :]
                batch_z_input = data_train_z[offset:(offset + batch_size), :]
                label_x_input = label_train_x[offset:(offset + batch_size)]
                label_y_input = label_train_y[offset:(offset + batch_size)]
                label_z_input = label_train_z[offset:(offset + batch_size)]

                # refine parameters
                sess.run(train_op,
                         feed_dict={x: batch_x_input,
                                    y: batch_y_input,
                                    z: batch_z_input,
                                    ref_label_x: label_x_input,
                                    ref_label_y: label_y_input,
                                    ref_label_z: label_z_input,
                                    triplet_lambda: lambda_super,
                                    triplet_margin: margin_estimate})

            # calculate loss on all input data for current epoch
            if epoch % print_epochs == 0:
                L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y, L_trip_z = \
                    sess.run((loss, reconstruction_error, weight_penalty, trip_loss_x, trip_loss_y, trip_loss_z),
                             feed_dict={x: data_train_x,
                                        y: data_train_y,
                                        z: data_train_z,
                                        ref_label_x: label_train_x,
                                        ref_label_y: label_train_y,
                                        ref_label_z: label_train_z,
                                        triplet_lambda: lambda_super,
                                        triplet_margin: margin_estimate})

                print(
                    "epoch: %d, \t total loss: %03.5f,\t reconstruction loss: %03.5f,\t sparse penalty: %03.5f,\t x triplet: %03.5f,\t y triplet: %03.5f,\t z triplet: %03.5f"
                    % (epoch, L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y, L_trip_z))

        # update cluster labels based modality-specific latents
        latent_x, latent_y, latent_z = \
            sess.run((encode_x, encode_y, encode_z),
                     feed_dict={x: data_x,
                                y: data_y,
                                z: data_z,
                                ref_label_x: label_x,
                                ref_label_y: label_y,
                                ref_label_z: label_z,
                                triplet_lambda: lambda_super,
                                triplet_margin: margin_estimate})

        # update cluster labels using PhenoGraph
        label_x_update, _, _ = phenograph.cluster(latent_x)
        label_y_update, _, _ = phenograph.cluster(latent_y)
        label_z_update, _, _ = phenograph.cluster(latent_z)
        print('Finish initialization of MUSE')

        ''' Training of MUSE '''
        for epoch in range(n_epochs):
            # randomly permute samples
            random_idx = np.random.permutation(n_sample)
            data_train_x = data_x[random_idx, :]
            data_train_y = data_y[random_idx, :]
            data_train_z = data_z[random_idx, :]
            label_train_x = label_x_update[random_idx]
            label_train_y = label_y_update[random_idx]
            label_train_z = label_z_update[random_idx]

            # loop over all batches
            for i in range(total_batch):
                # batch data
                offset = (i * batch_size) % (n_sample)
                batch_x_input = data_train_x[offset:(offset + batch_size), :]
                batch_y_input = data_train_y[offset:(offset + batch_size), :]
                batch_z_input = data_train_z[offset:(offset + batch_size), :]
                batch_label_x_input = label_train_x[offset:(offset + batch_size)]
                batch_label_y_input = label_train_y[offset:(offset + batch_size)]
                batch_label_z_input = label_train_z[offset:(offset + batch_size)]

                sess.run(train_op,
                         feed_dict={x: batch_x_input,
                                    y: batch_y_input,
                                    z: batch_z_input,
                                    ref_label_x: batch_label_x_input,
                                    ref_label_y: batch_label_y_input,
                                    ref_label_z: batch_label_z_input,
                                    triplet_lambda: lambda_super,
                                    triplet_margin: margin_estimate})

            # calculate and print losses on whole training dataset
            if epoch % print_epochs == 0:
                L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y, L_trip_z = \
                    sess.run((loss, reconstruction_error, weight_penalty, trip_loss_x, trip_loss_y, trip_loss_z),
                             feed_dict={x: data_train_x,
                                        y: data_train_y,
                                        z: data_train_z,
                                        ref_label_x: label_train_x,
                                        ref_label_y: label_train_y,
                                        ref_label_z: label_train_z,
                                        triplet_lambda: lambda_super,
                                        triplet_margin: margin_estimate})
                # print cost every epoch
                print(
                    "epoch: %d, \t total loss: %03.5f,\t reconstruction loss: %03.5f,\t sparse penalty: %03.5f,\t x triplet loss: %03.5f,\t y triplet loss: %03.5f,\t z triplet loss: %03.5f"
                    % (epoch, L_total, L_reconstruction, L_weight, L_trip_x, L_trip_y, L_trip_z))

            # update cluster labels based on new modality-specific latent representations
            if epoch % cluster_update_epoch == 0:
                latent_x, latent_y, latent_z = \
                    sess.run((encode_x, encode_y, encode_z),
                             feed_dict={x: data_x,
                                        y: data_y,
                                        z: data_z,
                                        ref_label_x: label_x,
                                        ref_label_y: label_y,
                                        ref_label_z: label_z,
                                        triplet_lambda: lambda_super,
                                        triplet_margin: margin_estimate})

                # use PhenoGraph to obtain cluster label
                label_x_update, _, _ = phenograph.cluster(latent_x)
                label_y_update, _, _ = phenograph.cluster(latent_y)
                label_z_update, _, _ = phenograph.cluster(latent_z)

        """ MUSE output """
        latent, reconstruct_x, reconstruct_y, reconstruct_z, latent_x, latent_y, latent_z = \
            sess.run((joint_latent, x_hat, y_hat, z_hat, encode_x, encode_y, encode_z),
                     feed_dict={x: data_x,
                                y: data_y,
                                z: data_z,
                                ref_label_x: label_x,  # no effects to representations
                                ref_label_y: label_y,  # no effects to representations
                                ref_label_z: label_z,  # no effects to representations
                                triplet_lambda: lambda_super,
                                triplet_margin: margin_estimate})

        print('++++++++++ MUSE completed ++++++++++')

    return latent, reconstruct_x, reconstruct_y, reconstruct_z, latent_x, latent_y, latent_z
