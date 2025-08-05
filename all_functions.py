# basic libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.utils import register_keras_serializable # type: ignore



# Tensorflow functions

#----------------------------------------------------------------- make_dataset---------------------------------------------------------------------------
def make_dataset(args, batch_size=128, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(args)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(args[0]))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

#----------------------------------------------------------------- plot_learning_curve---------------------------------------------------------------------------
def plot_learning_curve(history, save_path=None):
    logs = history.history
    keys = list(logs.keys())

    # train/val 쌍 추출
    paired_metrics = {}
    for key in keys:
        if key.startswith('val_'):
            base_key = key[4:]
            if base_key in logs:
                paired_metrics[base_key] = ('val_' + base_key, base_key)

    n = len(paired_metrics)
    include_lr = 'learning_rate' in logs
    total_plots = n + 1 if include_lr else n

    ncols = 3
    nrows = math.ceil(total_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.5 * nrows))
    axes = axes.flatten()

    # Metric plots
    for i, (metric, (val_key, train_key)) in enumerate(paired_metrics.items()):
        ax = axes[i]
        ax.plot(logs[train_key], label=f"train_{metric}")
        ax.plot(logs[val_key], label=f"val_{metric}")
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)

        if "total_loss" not in metric and \
            np.all(np.array(logs[train_key]) > 0) and \
            np.all(np.array(logs[val_key]) > 0):
            ax.set_yscale('log')

        ax.legend()
        ax.grid(True)

    # Learning rate plot
    if include_lr:
        ax = axes[n]
        ax.plot(logs['learning_rate'], label='learning_rate', color='tab:orange')
        ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)

    # 남은 subplot 제거
    for j in range(total_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

#----------------------------------------------------------------- Callbacks  ---------------------------------------------------------------------------

class EpochTracker(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.current_epoch.assign_add(1.0)


class LRSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            logs['learning_rate'] = lr

#----------------------------------------------------------------- Classes  ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class UncertaintyLoss(tf.keras.layers.Layer):
    def __init__(self, num_losses):
        super().__init__()
        self.log_vars = self.add_weight(name='log_vars', shape=(num_losses,), initializer='zeros', trainable=True)

    def call(self, loss_list):
        loss_list = tf.stack(loss_list)
        loss_list = tf.maximum(loss_list, 1e-12)
        weighted_losses = 0.5 * tf.exp(-self.log_vars) * loss_list + 0.5 * self.log_vars
        return tf.reduce_sum(weighted_losses)


@tf.keras.utils.register_keras_serializable()
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units, dropout=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout = dropout

        self.dense1 = layers.Dense(units, kernel_initializer='he_normal')
        self.norm1 = layers.LayerNormalization()
        self.act1 = layers.Activation('gelu')
        self.dense2 = layers.Dense(units, kernel_initializer='he_normal')
        self.norm2 = layers.LayerNormalization()
        self.final_act = layers.Activation('gelu')
        self.proj = None
        self.dropout_layer = layers.Dropout(dropout) if dropout else None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.proj = layers.Dense(self.units, kernel_initializer='he_normal')

    def call(self, inputs, training=False):
        shortcut = inputs
        x = self.dense1(inputs)
        x = self.norm1(x)
        x = self.act1(x)
        if self.dropout_layer:
            x = self.dropout_layer(x, training=training)
        x = self.dense2(x)
        x = self.norm2(x)
        if self.proj:
            shortcut = self.proj(shortcut)
        x = layers.Add()([x, shortcut])
        x = self.final_act(x)
        return x


#----------------------------------------------------------------- VAE ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class VAE(tf.keras.Model):
    def __init__(self, ds_dim, eels_dim, z_dim=20, **kwargs):
        super().__init__(**kwargs)
        self.ds_dim = ds_dim
        self.eels_dim = eels_dim
        self.z_dim = z_dim

        self.w_ds = 2.5
        self.w_eels = 1.0
        self.w_reg = 0.1
        self.alpha = 0.5

        self.current_epoch = tf.Variable(0, dtype=tf.float32, trainable=False)

        self.encoder = self.build_encoder(ds_dim, eels_dim, z_dim)
        self.decoder_ds = self.build_decoder_ds(ds_dim, z_dim)
        self.decoder_eels = self.build_decoder_eels(eels_dim, z_dim)



    # Encoder
    def build_encoder(self, ds_dim, eels_dim, z_dim):
        input_ds = layers.Input(shape=(ds_dim,), name='input_ds')
        input_eels = layers.Input(shape=(eels_dim,), name='input_eels')

        # Descriptor
        x = ResidualBlock(256, dropout=0.2)(input_ds)
        x = ResidualBlock(128)(x)
        x = ResidualBlock(64, dropout=0.2)(x)
        x = ResidualBlock(32)(x)

        # EELS
        y = ResidualBlock(512, dropout=0.2)(input_eels)
        y = ResidualBlock(256)(y)
        y = ResidualBlock(128, dropout=0.2)(y)
        y = ResidualBlock(64)(y)
        y = ResidualBlock(32)(y)

        # Merge
        merged = layers.Concatenate()([x, y])
        h = ResidualBlock(64)(merged)

        z_mean = layers.Dense(z_dim, name='z_mean')(h)
        z_logvar = layers.Dense(z_dim, name='z_logvar')(h)
        return tf.keras.Model(inputs=[input_ds, input_eels], outputs=[z_mean, z_logvar], name='Encoder')


    # Decoder for descriptor
    def build_decoder_ds(self, ds_dim, z_dim):
        input_z = layers.Input(shape=(z_dim,), name='input_z_ds')
        x = ResidualBlock(32, dropout=0.2)(input_z)
        x = ResidualBlock(64)(x)
        x = ResidualBlock(128, dropout=0.2)(x)
        x = ResidualBlock(256)(x)
        output_ds = layers.Dense(ds_dim, name='output_ds')(x)
        return tf.keras.Model(inputs=input_z, outputs=output_ds, name='Decoder_desc')


    # Decoder for EELS
    def build_decoder_eels(self, eels_dim, z_dim):
        input_z = layers.Input(shape=(z_dim,), name='input_z_eels')
        y = ResidualBlock(32, dropout=0.2)(input_z)
        y = ResidualBlock(64)(y)
        y = ResidualBlock(128, dropout=0.2)(y)
        y = ResidualBlock(256)(y)
        y = ResidualBlock(512)(y)
        output_eels = layers.Dense(eels_dim, name='output_eels')(y)
        return tf.keras.Model(inputs=input_z, outputs=output_eels, name='Decoder_EELS')


    # Sampling
    def sampling(self, z_mean, z_logvar):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon


    # Call
    def call(self, inputs, training=True):
        input_ds, input_eels = inputs
        z_mean, z_logvar = self.encoder([input_ds, input_eels], training=training)
        z_logvar = tf.clip_by_value(z_logvar, -20, 10)
        z = self.sampling(z_mean, z_logvar) if training else z_mean
        output_ds = self.decoder_ds(z, training=training)
        output_eels = self.decoder_eels(z, training=training)
        return output_ds, output_eels, z_mean, z_logvar, z


    def build(self, input_shape):
        super().build(input_shape)
        self.encoder.build([(None, self.ds_dim), (None, self.eels_dim)])
        self.decoder_ds.build((None, self.z_dim))
        self.decoder_eels.build((None, self.z_dim)) 
    

    def get_config(self):
        config = super().get_config()
        config.update({
            "ds_dim": self.ds_dim,
            "eels_dim": self.eels_dim,
            "z_dim": self.z_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def compute_ds_loss(self, x_true, x_pred):
        return tf.reduce_mean(tf.square(x_true - x_pred))

    def compute_eels_loss(self, y_true, y_pred):

        def pseudo_huber(x, delta=1.0):
            return delta**2 * (tf.sqrt(1.0 + tf.square(x / delta)) - 1.0)

        dy_true = y_true[:, 1:] - y_true[:, :-1]
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        ddy_true = dy_true[:, 1:] - dy_true[:, :-1]
        ddy_pred = dy_pred[:, 1:] - dy_pred[:, :-1]
        y_true_fft = tf.signal.fft(tf.cast(y_true, tf.complex64))
        y_pred_fft = tf.signal.fft(tf.cast(y_pred, tf.complex64))
        y_cumsum_true = tf.math.cumsum(y_true, axis=-1)
        y_cumsum_pred = tf.math.cumsum(y_pred, axis=-1)

        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        cosine_loss = (1.0 + tf.reduce_mean(tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)))
        slope_loss = tf.reduce_mean(pseudo_huber(dy_true - dy_pred, delta=0.05))
        curvature_loss = tf.reduce_mean(pseudo_huber(ddy_true - ddy_pred, delta=0.05))
        fft_loss = tf.reduce_mean(pseudo_huber(tf.abs(y_true_fft) - tf.abs(y_pred_fft), delta=0.05))
        cumsum_loss = tf.reduce_mean(pseudo_huber(y_cumsum_true - y_cumsum_pred, delta=0.05)) * 0.1
        roughness_loss = tf.reduce_mean(tf.square(dy_pred))

        eels_loss = mse_loss + cosine_loss + slope_loss + curvature_loss + fft_loss + cumsum_loss + roughness_loss
        return eels_loss
    
    def compute_kl_loss(self, z_mean, z_logvar):
        kl_per_dim = tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1
        kl_per_sample = 0.5 * tf.reduce_sum(kl_per_dim, axis=-1)
        return tf.reduce_mean(kl_per_sample)

    def compute_mmd_loss(self, z_sample, prior=None):
        if prior is None:
            prior = tf.random.normal(tf.shape(z_sample))
        diffs = tf.expand_dims(z_sample, 1) - tf.expand_dims(prior, 0)  # (N, N, D)
        pairwise_dists = tf.reduce_sum(tf.square(diffs), axis=-1)  # (N, N)
        flat = tf.reshape(pairwise_dists, [-1])
        median_dist = tf.sort(flat)[tf.shape(flat)[0] // 2]
        kernel_width = tf.maximum(tf.stop_gradient(median_dist), 1e-6)

        def rbf_kernel(x, y, kernel_width):
            x_expand = tf.expand_dims(x, 1)
            y_expand = tf.expand_dims(y, 0)
            sq_dist = tf.reduce_sum(tf.square(x_expand - y_expand), axis=-1)
            return tf.exp(-sq_dist / (2.0 * kernel_width))

        K_xx = rbf_kernel(z_sample, z_sample, kernel_width)
        K_yy = rbf_kernel(prior, prior, kernel_width)
        K_xy = rbf_kernel(z_sample, prior, kernel_width)
        return tf.reduce_mean(K_xx + K_yy - 2 * K_xy)

    def get_annealed_weight(self, anneal_epoch, start_epoch=0):
        progress = tf.maximum(0.0, tf.cast(self.current_epoch - start_epoch, tf.float32) / tf.cast(tf.maximum(anneal_epoch, 1), tf.float32))
        return tf.minimum(1.0, progress)


    # VAE loss
    def VAE_loss(self, x, y):
        x_pred, y_pred, z_mean, z_logvar, z = self.call([x, y])

        # Losses
        ds_loss = self.w_ds * self.compute_ds_loss(x, x_pred)
        eels_loss = self.w_eels * self.compute_eels_loss(y, y_pred)
        kl_loss = self.compute_kl_loss(z_mean, z_logvar)
        mmd_loss = self.compute_mmd_loss(z)
        w_reg = self.w_reg * self.get_annealed_weight(70, start_epoch=30)
        reg_loss = w_reg * (self.alpha * kl_loss + (1 - self.alpha) * mmd_loss)

        # Total loss
        total_loss =  ds_loss + eels_loss + reg_loss

        return total_loss, {
            "Loss_DS": ds_loss,
            "Loss_EELS": eels_loss,
            "Loss_REG": reg_loss,
            "MSE_DS": tf.reduce_mean(tf.square(x - x_pred)),
            "MSE_EELS": tf.reduce_mean(tf.square(y - y_pred)),
            "Loss_RECON": ds_loss + eels_loss,
        }

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            loss, loss_dict = self.VAE_loss(x, y)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"total_loss": loss, **loss_dict}

    @tf.function
    def test_step(self, data):
        x, y = data
        loss, loss_dict = self.VAE_loss(x, y)
        return {"total_loss": loss, **loss_dict}





#----------------------------------------------------------------- VAE ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class Inverse_Model(tf.keras.Model):
    def __init__(self, ds_dim, eels_dim, z_dim=20, VAE_model=None, **kwargs):
        super().__init__(**kwargs)
        self.ds_dim = ds_dim
        self.eels_dim = eels_dim
        self.z_dim = z_dim
        self.vae = VAE_model
        if self.vae is not None:
            self.vae.trainable = False
            self.z_dim = self.vae.z_dim

        self.w_z = 1.0
        self.w_ds = 5.0
        self.w_eels = 1.0

        self.encoder_inverse = self.build_encoder_inverse(ds_dim, eels_dim, z_dim)


    # EELS Encoder
    def build_encoder_inverse(self, ds_dim, eels_dim, z_dim):
        input_eels = layers.Input(shape=(eels_dim,), name='input_eels')
        y = ResidualBlock(256, dropout=0.3)(input_eels)
        y = ResidualBlock(64, dropout=0.3)(y)
        z_mean = layers.Dense(z_dim, name='z_mean')(y)
        return tf.keras.Model(inputs=input_eels, outputs=z_mean, name='Encoder_Inverse')


    # Call
    def call(self, inputs, training=True):
        z_pred = self.encoder_inverse(inputs, training=training)
        return z_pred


    def set_vae(self, vae_model):
        object.__setattr__(self, "vae", vae_model)
        self.vae.trainable = False

    def build(self, input_shape):
        super().build(input_shape)
        self.encoder_inverse.build((None, self.eels_dim))


    def get_config(self):
        config = super().get_config()
        config.update({
            "ds_dim": self.ds_dim,
            "eels_dim": self.eels_dim,
            "z_dim": self.z_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def compute_ds_loss(self, x_true, x_pred): 
        return tf.reduce_mean(tf.square(x_true - x_pred))

    def compute_eels_loss(self, y_true, y_pred):

        def pseudo_huber(x, delta=1.0):
            return delta**2 * (tf.sqrt(1.0 + tf.square(x / delta)) - 1.0)

        dy_true = y_true[:, 1:] - y_true[:, :-1]
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        ddy_true = dy_true[:, 1:] - dy_true[:, :-1]
        ddy_pred = dy_pred[:, 1:] - dy_pred[:, :-1]
        y_true_fft = tf.signal.fft(tf.cast(y_true, tf.complex64))
        y_pred_fft = tf.signal.fft(tf.cast(y_pred, tf.complex64))
        y_cumsum_true = tf.math.cumsum(y_true, axis=-1)
        y_cumsum_pred = tf.math.cumsum(y_pred, axis=-1)

        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        cosine_loss = (1.0 + tf.reduce_mean(tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)))
        slope_loss = tf.reduce_mean(pseudo_huber(dy_true - dy_pred, delta=0.05))
        curvature_loss = tf.reduce_mean(pseudo_huber(ddy_true - ddy_pred, delta=0.05))
        fft_loss = tf.reduce_mean(pseudo_huber(tf.abs(y_true_fft) - tf.abs(y_pred_fft), delta=0.05))
        cumsum_loss = tf.reduce_mean(pseudo_huber(y_cumsum_true - y_cumsum_pred, delta=0.05)) * 0.1
        roughness_loss = tf.reduce_mean(tf.square(dy_pred))

        eels_loss = mse_loss + cosine_loss + slope_loss + curvature_loss + fft_loss + cumsum_loss + roughness_loss
        return eels_loss
    

    # VAE loss
    def Inverse_Model_loss(self, y, x):
        z_pred = self.call(y)
        z_true, _ = self.vae.encoder([x, y], training=False)
        x_pred = self.vae.decoder_ds(z_pred, training=False)
        y_pred = self.vae.decoder_eels(z_pred, training=False)

        # Losses
        z_loss = self.w_z * self.compute_ds_loss(z_true, z_pred)
        ds_loss = self.w_ds * self.compute_ds_loss(x, x_pred)
        eels_loss = self.w_eels * self.compute_eels_loss(y, y_pred)

        # Total loss
        total_loss =  z_loss + ds_loss + eels_loss
    
        return total_loss, {
            "MSE_Z": z_loss / self.w_z,
            "MSE_DS": ds_loss / self.w_ds,
            "MSE_EELS": self.compute_ds_loss(y, y_pred),
        }

    @tf.function
    def train_step(self, data):
        y, x = data
        with tf.GradientTape() as tape:
            loss, loss_dict = self.Inverse_Model_loss(y, x)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"total_loss": loss, **loss_dict}

    @tf.function
    def test_step(self, data):
        y, x = data
        loss, loss_dict = self.Inverse_Model_loss(y, x)
        return {"total_loss": loss, **loss_dict}



















#----------------------------------------------------------------- VAE ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class Inverse_Model_MLP(tf.keras.Model):
    def __init__(self, ds_dim, eels_dim, **kwargs):
        super().__init__(**kwargs)
        self.ds_dim = ds_dim
        self.eels_dim = eels_dim
        self.w_ds = 1.0
        self.mlp = self.build_mlp(ds_dim, eels_dim)


    # EELS Encoder
    def build_mlp(self, ds_dim, eels_dim):
        input_eels = layers.Input(shape=(eels_dim,), name='input_eels')
        y = ResidualBlock(1024)(input_eels)
        y = ResidualBlock(512)(y)
        y = ResidualBlock(128)(y)
        output_ds = layers.Dense(ds_dim, name='output_ds')(y)
        return tf.keras.Model(inputs=input_eels, outputs=output_ds, name='MLP')


    # Call
    def call(self, inputs):
        x_pred = self.mlp(inputs)
        return x_pred


    def build(self, input_shape):
        super().build(input_shape)
        self.mlp.build((None, self.eels_dim))
    

    def get_config(self):
        config = super().get_config()
        config.update({
            "ds_dim": self.ds_dim,
            "eels_dim": self.eels_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def compute_ds_loss(self, x_true, x_pred): 
        return tf.reduce_mean(tf.square(x_true - x_pred))

    # VAE loss
    def Inverse_Model_loss(self, y, x):
        x_pred = self.call(y)

        # Losses
        ds_loss = self.w_ds * self.compute_ds_loss(x, x_pred)

        # Total loss
        total_loss =  ds_loss
    
        return total_loss, {
            "Loss_DS": ds_loss,
            # "MSE_DS": tf.reduce_mean(tf.square(x - x_pred)),
        }

    @tf.function
    def train_step(self, data):
        y, x = data
        with tf.GradientTape() as tape:
            loss, loss_dict = self.Inverse_Model_loss(y, x)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"total_loss": loss, **loss_dict}

    @tf.function
    def test_step(self, data):
        y, x = data
        loss, loss_dict = self.Inverse_Model_loss(y, x)
        return {"total_loss": loss, **loss_dict}





#----------------------------------------------------------------- VAE ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class VAE_nll(tf.keras.Model):
    def __init__(self, ds_dim, eels_dim, z_dim=9, **kwargs):
        super().__init__(**kwargs)
        self.ds_dim = ds_dim
        self.eels_dim = eels_dim
        self.z_dim = z_dim

        self.w_ds = 2.0
        self.w_eels = 1.0
        self.w_reg = 0.1
        self.alpha = 0.5

        self.current_epoch = tf.Variable(0, dtype=tf.float32, trainable=False)

        self.encoder = self.build_encoder(ds_dim, eels_dim, z_dim)
        self.decoder_ds = self.build_decoder_ds(ds_dim, z_dim)
        self.decoder_eels = self.build_decoder_eels(eels_dim, z_dim)



    # Encoder
    def build_encoder(self, ds_dim, eels_dim, z_dim):
        input_ds = layers.Input(shape=(ds_dim,), name='input_ds')
        input_eels = layers.Input(shape=(eels_dim,), name='input_eels')

        # Descriptor
        x = ResidualBlock(128)(input_ds)
        x = ResidualBlock(64, dropout=0.2)(x)
        x = ResidualBlock(16)(x)

        # EELS
        y = ResidualBlock(512)(input_eels)
        y = ResidualBlock(128)(y)
        y = ResidualBlock(64, dropout=0.2)(y)
        y = ResidualBlock(16)(y)

        # Merge
        merged = layers.Concatenate()([x, y])
        h = ResidualBlock(32)(merged)

        z_mean = layers.Dense(z_dim, name='z_mean')(h)
        z_logvar = layers.Dense(z_dim, name='z_logvar')(h)
        return tf.keras.Model(inputs=[input_ds, input_eels], outputs=[z_mean, z_logvar], name='Encoder')


    # Decoder for descriptor
    def build_decoder_ds(self, ds_dim, z_dim):
        input_z = layers.Input(shape=(z_dim,), name='input_z_ds')
        x = ResidualBlock(16)(input_z)
        x = ResidualBlock(32)(x)

        h = ResidualBlock(64)(x)
        h = ResidualBlock(128)(h)
        h = ResidualBlock(256)(h)
        h = ResidualBlock(512)(h)

        j = ResidualBlock(64)(x)

        x_pred = layers.Dense(ds_dim, name='x_pred')(h)
        x_logvar = layers.Dense(ds_dim, name='x_logvar')(j)
        return tf.keras.Model(inputs=input_z, outputs=[x_pred, x_logvar], name='Decoder_desc')


    # Decoder for EELS
    def build_decoder_eels(self, eels_dim, z_dim):
        input_z = layers.Input(shape=(z_dim,), name='input_z_eels')
        y = ResidualBlock(32)(input_z)
        y = ResidualBlock(64, dropout=0.5)(y)
        y = ResidualBlock(128)(y)
        y = ResidualBlock(512)(y)
        output_eels = layers.Dense(eels_dim, name='output_eels')(y)
        return tf.keras.Model(inputs=input_z, outputs=output_eels, name='Decoder_EELS')


    # Sampling
    def sampling(self, z_mean, z_logvar):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon


    # Call
    def call(self, inputs, training=True):
        input_ds, input_eels = inputs
        z_mean, z_logvar = self.encoder([input_ds, input_eels])
        z_logvar = tf.clip_by_value(z_logvar, -20, 10)
        z = self.sampling(z_mean, z_logvar) if training else z_mean
        x_pred, x_logvar = self.decoder_ds(z)
        x_logvar = tf.clip_by_value(x_logvar, -20, 10)
        output_eels = self.decoder_eels(z)
        return x_pred, x_logvar, output_eels, z_mean, z_logvar, z


    def build(self, input_shape):
        super().build(input_shape)
        self.encoder.build([(None, self.ds_dim), (None, self.eels_dim)])
        self.decoder_ds.build((None, self.z_dim))
        self.decoder_eels.build((None, self.z_dim)) 
    

    def get_config(self):
        config = super().get_config()
        config.update({
            "ds_dim": self.ds_dim,
            "eels_dim": self.eels_dim,
            "z_dim": self.z_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def compute_ds_loss(self, x_true, x_pred, x_logvar):
        nll = 0.5 * ((tf.square(x_true - x_pred) / tf.exp(x_logvar)) + x_logvar)
        nll_loss = tf.reduce_mean(tf.reduce_sum(nll, axis=-1))
        return nll_loss

    def compute_eels_loss(self, y_true, y_pred):

        def pseudo_huber(x, delta=1.0):
            return delta**2 * (tf.sqrt(1.0 + tf.square(x / delta)) - 1.0)

        dy_true = y_true[:, 1:] - y_true[:, :-1]
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        ddy_true = dy_true[:, 1:] - dy_true[:, :-1]
        ddy_pred = dy_pred[:, 1:] - dy_pred[:, :-1]
        y_true_fft = tf.signal.fft(tf.cast(y_true, tf.complex64))
        y_pred_fft = tf.signal.fft(tf.cast(y_pred, tf.complex64))
        y_cumsum_true = tf.math.cumsum(y_true, axis=-1)
        y_cumsum_pred = tf.math.cumsum(y_pred, axis=-1)

        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        cosine_loss = (1.0 + tf.reduce_mean(tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)))
        slope_loss = tf.reduce_mean(pseudo_huber(dy_true - dy_pred, delta=0.05))
        curvature_loss = tf.reduce_mean(pseudo_huber(ddy_true - ddy_pred, delta=0.05))
        fft_loss = tf.reduce_mean(pseudo_huber(tf.abs(y_true_fft) - tf.abs(y_pred_fft), delta=0.05))
        cumsum_loss = tf.reduce_mean(pseudo_huber(y_cumsum_true - y_cumsum_pred, delta=0.05)) * 0.1
        roughness_loss = tf.reduce_mean(tf.square(dy_pred))

        eels_loss = mse_loss + cosine_loss + slope_loss + curvature_loss + fft_loss + cumsum_loss + roughness_loss
        return eels_loss
    
    def compute_kl_loss(self, z_mean, z_logvar):
        kl_per_dim = tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1
        kl_per_sample = 0.5 * tf.reduce_sum(kl_per_dim, axis=-1)
        return tf.reduce_mean(kl_per_sample)

    def compute_mmd_loss(self, z_sample, prior=None):
        if prior is None:
            prior = tf.random.normal(tf.shape(z_sample))
        diffs = tf.expand_dims(z_sample, 1) - tf.expand_dims(prior, 0)  # (N, N, D)
        pairwise_dists = tf.reduce_sum(tf.square(diffs), axis=-1)  # (N, N)
        flat = tf.reshape(pairwise_dists, [-1])
        median_dist = tf.sort(flat)[tf.shape(flat)[0] // 2]
        kernel_width = tf.maximum(tf.stop_gradient(median_dist), 1e-6)

        def rbf_kernel(x, y, kernel_width):
            x_expand = tf.expand_dims(x, 1)
            y_expand = tf.expand_dims(y, 0)
            sq_dist = tf.reduce_sum(tf.square(x_expand - y_expand), axis=-1)
            return tf.exp(-sq_dist / (2.0 * kernel_width))

        K_xx = rbf_kernel(z_sample, z_sample, kernel_width)
        K_yy = rbf_kernel(prior, prior, kernel_width)
        K_xy = rbf_kernel(z_sample, prior, kernel_width)
        return tf.reduce_mean(K_xx + K_yy - 2 * K_xy)

    def get_annealed_weight(self, anneal_epoch, start_epoch=0):
        progress = tf.maximum(0.0, tf.cast(self.current_epoch - start_epoch, tf.float32) / tf.cast(tf.maximum(anneal_epoch, 1), tf.float32))
        return tf.minimum(1.0, progress)


    # VAE loss
    def VAE_loss(self, x, y):
        x_pred, x_logvar, y_pred, z_mean, z_logvar, z = self.call([x, y])

        # Losses
        ds_loss = self.w_ds * self.compute_ds_loss(x, x_pred, x_logvar)
        eels_loss = self.w_eels * self.compute_eels_loss(y, y_pred)
        kl_loss = self.compute_kl_loss(z_mean, z_logvar)
        mmd_loss = self.compute_mmd_loss(z)
        w_reg = self.w_reg * self.get_annealed_weight(70, start_epoch=30)
        reg_loss = w_reg * (self.alpha * kl_loss + (1 - self.alpha) * mmd_loss)

        # Total loss
        total_loss =  ds_loss + eels_loss + reg_loss

        return total_loss, {
            "Loss_DS": ds_loss,
            "Loss_EELS": eels_loss,
            "Loss_REG": reg_loss,
            "MSE_DS": tf.reduce_mean(tf.square(x - x_pred)),
            "MSE_EELS": tf.reduce_mean(tf.square(y - y_pred)),
            "Loss_RECON": ds_loss + eels_loss,
        }

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            loss, loss_dict = self.VAE_loss(x, y)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"total_loss": loss, **loss_dict}

    @tf.function
    def test_step(self, data):
        x, y = data
        loss, loss_dict = self.VAE_loss(x, y)
        return {"total_loss": loss, **loss_dict}




#----------------------------------------------------------------- VAE ---------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class VAE_not_merge(tf.keras.Model):
    def __init__(self, ds_dim, eels_dim, z_dim=9, **kwargs):
        super().__init__(**kwargs)
        self.ds_dim = ds_dim
        self.eels_dim = eels_dim
        self.z_dim = z_dim

        self.w_ds = 2.0
        self.w_eels = 1.0
        self.w_reg = 0.1
        self.alpha = 0.5

        self.current_epoch = tf.Variable(0, dtype=tf.float32, trainable=False)

        self.encoder = self.build_encoder(ds_dim, z_dim)
        self.decoder_ds = self.build_decoder_ds(ds_dim, z_dim)
        self.decoder_eels = self.build_decoder_eels(eels_dim, z_dim)



    # Encoder
    def build_encoder(self, ds_dim, z_dim):
        input_ds = layers.Input(shape=(ds_dim,), name='input_ds')
        x = ResidualBlock(128)(input_ds)
        x = ResidualBlock(64)(x)
        x = ResidualBlock(32)(x)
        z_mean = layers.Dense(z_dim, name='z_mean')(x)
        z_logvar = layers.Dense(z_dim, name='z_logvar')(x)
        return tf.keras.Model(inputs=input_ds, outputs=[z_mean, z_logvar], name='Encoder')


    # Decoder for descriptor
    def build_decoder_ds(self, ds_dim, z_dim):
        input_z = layers.Input(shape=(z_dim,), name='input_z_ds')
        x = ResidualBlock(16)(input_z)
        x = ResidualBlock(64)(x)
        x = ResidualBlock(128)(x)
        output_ds = layers.Dense(ds_dim, name='output_ds')(x)
        return tf.keras.Model(inputs=input_z, outputs=output_ds, name='Decoder_desc')


    # Decoder for EELS
    def build_decoder_eels(self, eels_dim, z_dim):
        input_z = layers.Input(shape=(z_dim,), name='input_z_eels')
        y = ResidualBlock(16)(input_z)
        y = ResidualBlock(64)(y)
        y = ResidualBlock(128)(y)
        y = ResidualBlock(512)(y)
        output_eels = layers.Dense(eels_dim, name='output_eels')(y)
        return tf.keras.Model(inputs=input_z, outputs=output_eels, name='Decoder_EELS')


    # Sampling
    def sampling(self, z_mean, z_logvar):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon


    # Call
    def call(self, input_ds, training=True):
        z_mean, z_logvar = self.encoder(input_ds)
        z_logvar = tf.clip_by_value(z_logvar, -20, 10)
        z = self.sampling(z_mean, z_logvar) if training else z_mean
        output_ds = self.decoder_ds(z)
        output_eels = self.decoder_eels(z)
        return output_ds, output_eels, z_mean, z_logvar, z


    def build(self, input_shape):
        super().build(input_shape)
        self.encoder.build((None, self.ds_dim))
        self.decoder_ds.build((None, self.z_dim))
        self.decoder_eels.build((None, self.z_dim)) 
    

    def get_config(self):
        config = super().get_config()
        config.update({
            "ds_dim": self.ds_dim,
            "eels_dim": self.eels_dim,
            "z_dim": self.z_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def compute_ds_loss(self, x_true, x_pred):
        return tf.reduce_mean(tf.square(x_true - x_pred))

    def compute_eels_loss(self, y_true, y_pred):

        def pseudo_huber(x, delta=1.0):
            return delta**2 * (tf.sqrt(1.0 + tf.square(x / delta)) - 1.0)

        dy_true = y_true[:, 1:] - y_true[:, :-1]
        dy_pred = y_pred[:, 1:] - y_pred[:, :-1]
        ddy_true = dy_true[:, 1:] - dy_true[:, :-1]
        ddy_pred = dy_pred[:, 1:] - dy_pred[:, :-1]
        y_true_fft = tf.signal.fft(tf.cast(y_true, tf.complex64))
        y_pred_fft = tf.signal.fft(tf.cast(y_pred, tf.complex64))
        y_cumsum_true = tf.math.cumsum(y_true, axis=-1)
        y_cumsum_pred = tf.math.cumsum(y_pred, axis=-1)

        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        cosine_loss = (1.0 + tf.reduce_mean(tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1))) * 0.1
        slope_loss = tf.reduce_mean(pseudo_huber(dy_true - dy_pred, delta=0.05))
        curvature_loss = tf.reduce_mean(pseudo_huber(ddy_true - ddy_pred, delta=0.05))
        fft_loss = tf.reduce_mean(pseudo_huber(tf.abs(y_true_fft) - tf.abs(y_pred_fft), delta=0.05))
        cumsum_loss = tf.reduce_mean(pseudo_huber(y_cumsum_true - y_cumsum_pred, delta=0.05)) * 0.1
        roughness_loss = tf.reduce_mean(tf.square(dy_pred)) * 1.0

        eels_loss = mse_loss + cosine_loss + slope_loss + curvature_loss + fft_loss + cumsum_loss + roughness_loss
        return eels_loss
    
    def compute_kl_loss(self, z_mean, z_logvar):
        kl_per_dim = tf.square(z_mean) + tf.exp(z_logvar) - z_logvar - 1
        kl_per_sample = 0.5 * tf.reduce_sum(kl_per_dim, axis=-1)
        return tf.reduce_mean(kl_per_sample)

    def compute_mmd_loss(self, z_sample, prior=None):
        if prior is None:
            prior = tf.random.normal(tf.shape(z_sample))
        diffs = tf.expand_dims(z_sample, 1) - tf.expand_dims(prior, 0)  # (N, N, D)
        pairwise_dists = tf.reduce_sum(tf.square(diffs), axis=-1)  # (N, N)
        flat = tf.reshape(pairwise_dists, [-1])
        median_dist = tf.sort(flat)[tf.shape(flat)[0] // 2]
        kernel_width = tf.maximum(tf.stop_gradient(median_dist), 1e-6)

        def rbf_kernel(x, y, kernel_width):
            x_expand = tf.expand_dims(x, 1)
            y_expand = tf.expand_dims(y, 0)
            sq_dist = tf.reduce_sum(tf.square(x_expand - y_expand), axis=-1)
            return tf.exp(-sq_dist / (2.0 * kernel_width))

        K_xx = rbf_kernel(z_sample, z_sample, kernel_width)
        K_yy = rbf_kernel(prior, prior, kernel_width)
        K_xy = rbf_kernel(z_sample, prior, kernel_width)
        return tf.reduce_mean(K_xx + K_yy - 2 * K_xy)

    def get_annealed_weight(self, anneal_epoch, start_epoch=0):
        progress = tf.maximum(0.0, tf.cast(self.current_epoch - start_epoch, tf.float32) / tf.cast(tf.maximum(anneal_epoch, 1), tf.float32))
        return tf.minimum(1.0, progress)


    # VAE loss
    def VAE_loss(self, x, y):
        x_pred, y_pred, z_mean, z_logvar, z = self.call(x)

        # Losses
        ds_loss = self.w_ds * self.compute_ds_loss(x, x_pred)
        eels_loss = self.w_eels * self.compute_eels_loss(y, y_pred)
        kl_loss = self.compute_kl_loss(z_mean, z_logvar)
        mmd_loss = self.compute_mmd_loss(z)
        w_reg = self.w_reg * self.get_annealed_weight(70, start_epoch=30)
        reg_loss = w_reg * (self.alpha * kl_loss + (1 - self.alpha) * mmd_loss)

        # Total loss
        total_loss =  ds_loss + eels_loss + reg_loss

        return total_loss, {
            "Loss_DS": ds_loss,
            "Loss_EELS": eels_loss,
            "Loss_REG": reg_loss,
            "MSE_DS": tf.reduce_mean(tf.square(x - x_pred)),
            "MSE_EELS": tf.reduce_mean(tf.square(y - y_pred)),
            "Loss_RECON": ds_loss + eels_loss,
        }

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            loss, loss_dict = self.VAE_loss(x, y)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"total_loss": loss, **loss_dict}

    @tf.function
    def test_step(self, data):
        x, y = data
        loss, loss_dict = self.VAE_loss(x, y)
        return {"total_loss": loss, **loss_dict}



