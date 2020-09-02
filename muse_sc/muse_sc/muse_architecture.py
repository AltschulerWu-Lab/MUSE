import tensorflow as tf
from .triplet_loss import batch_hard_triplet_loss

""" model structure of MUSE """


def structured_embedding(x, y, label_x, label_y, dim_z, triplet_margin,
                         n_hidden, weight_penalty, triplet_lambda):
    """
    Construct structure and loss function of MUSE

    Parameters:
        x:              input batches for transcript modality; matrix of  n * p, where n = batch size, p = number of genes
        y:              input batches for morphological modality; matrix of n * q, where n = batch size, q is the feature dimension
        label_x:        input sample labels for modality x
        label_y:        input sample labels for modality y
        dim_z:          dimension of joint latent representation
        triplet_margin: margin for triplet loss
        n_hidden:       hidden node number for encoder and decoder layers
        weight_penalty: weight for sparse penalty
        triplet_lambda: weight for triplet loss

    Outputs:
        z:              joint latent representations (n * dim_z)
        x_hat:          reconstructed x (same shape as x)
        y_hat:          reconstructed y (same shape as y)
        encode_x:       latent representation for modality x
        encode_y:       latent representation for modality y
        loss:           total loss
        reconstruct_loss: reconstruction loss
        sparse_penalty: sparse penalty
        trip_loss_x:    triplet loss for x
        trip_loss_y:    triplet loss for y

    Altschuler & Wu Lab 2020.
    Software provided as is under MIT License.
    """

    # encoder
    z, encode_x, encode_y = multiview_encoder(x, y, n_hidden, dim_z)

    # decoder
    w_init = tf.initializers.random_normal()
    w_x = tf.get_variable('w_selection_x', [z.get_shape()[1], z.get_shape()[1]], initializer=w_init)
    w_y = tf.get_variable('w_selection_y', [z.get_shape()[1], z.get_shape()[1]], initializer=w_init)

    with tf.variable_scope("decoder_x"):
        z_x = tf.matmul(z, w_x)
        x_hat = decoder(z_x, n_hidden, x.get_shape()[1])

    with tf.variable_scope("decoder_y"):
        z_y = tf.matmul(z, w_y)
        y_hat = decoder(z_y, n_hidden, y.get_shape()[1])

    # sparse penalty
    sparse_x = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(w_x), axis=1)))
    sparse_y = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.square(w_y), axis=1)))
    sparse_penalty = sparse_x + sparse_y

    # reconstruction errors (for x modality, only non-zero entries were used)
    x_mask = tf.sign(x)
    reconstruct_x = tf.reduce_sum(tf.norm(tf.multiply(x_mask, (x_hat - x)))) / tf.reduce_sum(x_mask)
    reconstruct_y = tf.reduce_mean(tf.norm(y_hat - y))
    reconstruct_loss = reconstruct_x + reconstruct_y

    # triplet errors
    trip_loss_x = batch_hard_triplet_loss(label_x, z, triplet_margin)
    trip_loss_y = batch_hard_triplet_loss(label_y, z, triplet_margin)

    loss = reconstruct_loss + weight_penalty * sparse_penalty \
           + triplet_lambda * trip_loss_x \
           + triplet_lambda * trip_loss_y

    return z, x_hat, y_hat, encode_x, encode_y, \
           loss, reconstruct_loss, sparse_penalty, trip_loss_x, trip_loss_y


def multiview_encoder(x, y, n_hidden, dim_z):
    """
    Encoder combines x and y to a joint latent representation

    Parameters:
        x:              input batches for transcript modality; matrix of  n * p, where n = batch size, p = number of genes
        y:              input batches for morphological modality; matrix of n * q, where n = batch size, q is the feature dimension
        n_hidden:       hidden node number for encoder and decoder layers
        dim_z:          dimension of joint latent representations

    Outputs:
        latent:         joint latent representations
        h_x:            latent representation for modality x
        h_y:            latent representation for modality y

    Altschuler & Wu Lab 2020.
    Software provided as is under MIT License.
    """

    # encoder for each modality
    with tf.variable_scope("encoder_x"):
        h_x = encoder(x, n_hidden)
    with tf.variable_scope("encoder_y"):
        h_y = encoder(y, n_hidden)

    # combine h_x and h_y to joint latent representation
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)
    h = tf.concat([h_x, h_y], 1)
    wo = tf.get_variable('wo', [h.get_shape()[1], dim_z], initializer=w_init)
    bo = tf.get_variable('bo', [dim_z], initializer=b_init)
    latent = tf.matmul(h, wo) + bo

    return latent, h_x, h_y


def encoder(x, n_hidden):
    """
    Encoder for single modality

    Parameters:
        x:              input batches of single modality
        n_hidden:       hidden node number

    Outputs:
        o:              latent representation for single modality

    Altschuler & Wu Lab 2020.
    Software provided as is under MIT License.
    """

    # initializers
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('w0_e', [x.get_shape()[1], n_hidden], initializer=w_init)
    b0 = tf.get_variable('b0_e', [n_hidden], initializer=b_init)
    h0 = tf.matmul(x, w0) + b0
    h0 = tf.nn.elu(h0)

    # 2nd hidden layer
    w1 = tf.get_variable('w1_e', [h0.get_shape()[1], n_hidden], initializer=w_init)
    b1 = tf.get_variable('b1_e', [n_hidden], initializer=b_init)
    h1 = tf.matmul(h0, w1) + b1
    o = tf.nn.tanh(h1)

    return o


def decoder(z, n_hidden, n_output):
    """
    Decoder for single modality

    Parameters:
        z:              latent representation for single modality
        n_hidden:       hidden node number in decoder
        n_output:       feature dimension of original data

    Outputs:
        y:              reconstructed feature

    Altschuler & Wu Lab 2020.
    Software provided as is under MIT License.
    """

    # initializers
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(0.)

    # 1st hidden layer
    w0 = tf.get_variable('w0_d', [z.get_shape()[1], n_hidden], initializer=w_init)
    b0 = tf.get_variable('b0_d', [n_hidden], initializer=b_init)
    h0 = tf.matmul(z, w0) + b0
    h0 = tf.nn.elu(h0)

    # 2nd hidden layer
    w1 = tf.get_variable('w1_d', [h0.get_shape()[1], n_hidden], initializer=w_init)
    b1 = tf.get_variable('b1_d', [n_hidden], initializer=b_init)
    h1 = tf.matmul(h0, w1) + b1
    h1 = tf.nn.tanh(h1)

    # output layer-mean
    wo = tf.get_variable('wo_d', [h1.get_shape()[1], n_output], initializer=w_init)
    bo = tf.get_variable('bo_d', [n_output], initializer=b_init)
    y = tf.matmul(h1, wo) + bo

    return y
