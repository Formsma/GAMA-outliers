import numpy as np

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda, Dropout
from tensorflow.keras.losses import binary_crossentropy, mse
from tensorflow.keras.optimizers import Adam


class InfoVAE():
    """InfoVAE is a variational autoencoder method defined to mitigate
    a few problems of traditional vae's and improve the results. This 
    is done by redifining the loss function to better reparametrize
    the data. On top of the reconstruction loss we have two additional
    parameters 'KL divergence' and an 'MMD' term.
    
    Original paper: 
      https://arxiv.org/abs/1706.02262
    Paper used for this class: 
      https://arxiv.org/abs/2002.10464
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim, alpha=0, lambd=1):
        """Initialization of the InfoVAE needs input parameters
        for a normal VAE.
        
        input:
            input_dim:    input and output dimension (int)
            hidden_dim:   dimension of hidden layer (list of length 2)
            latent_dim:   latend dimension
            alpha:        KL Divergence suppression
            lambd:        MMD Divergence activation
        """
        
        # Set full VAE to empty for now
        self.vae = None
    
        # Store VAE properties
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Loss variables
        self.alpha = alpha
        self.lambd = lambd
        
        # Initialize VAE
        self._init_vae()
        
    def _compute_kernel(self, x, y):
        """Compute momentum of samples x and y via Gaussian kernels"""
        batch = K.shape(x)[0]
        
        tiled_x = K.tile(K.reshape(x, K.stack([batch, 1, self.latent_dim])), K.stack([1, batch, 1]))
        tiled_y = K.tile(K.reshape(y, K.stack([1, batch, self.latent_dim])), K.stack([batch, 1, 1]))
        
        return K.exp(-K.mean(K.square(tiled_x-tiled_y), axis=2) / self.latent_dim)
        
    def _compute_mmd(self, x, y):
        """Compute MMD Divergence on samples x and y"""
        xx_kernel = self._compute_kernel(x, x)
        xy_kernel = self._compute_kernel(x, y)
        yy_kernel = self._compute_kernel(y, y)
        return K.mean(xx_kernel) + K.mean(yy_kernel) - 2 * K.mean(xy_kernel)
        
    def _init_vae(self):
        """Initialize the variational autoencoder"""
        
        # Encoding layers
        inputs = Input(shape=(self.input_dim,), name='raw_input')  
        h_encoder_0 = Dense(self.hidden_dim[0], activation='relu')(inputs)
        
        for l in self.hidden_dim[1:]:
            h_encoder_0 = Dense(l, activation='relu')(h_encoder_0)
        
        # Latent layers
        self.z_mean = Dense(self.latent_dim, name='z_mean')(h_encoder_0)
        self.z_log_var = Dense(self.latent_dim, name='z_log_var')(h_encoder_0)
        
        # Sample distribution from latent layers
        self.z = Lambda(self._sampler)([self.z_mean, self.z_log_var])
        
        # Define encoder model
        self.encoder = Model(inputs, [self.z_mean, self.z_log_var, self.z], name='Encoder')
        
        # Define decoder model
        self.decoder = Sequential(name='Decoder')
        self.decoder.add(Dense(self.hidden_dim[-1], input_shape=(self.latent_dim,), activation='relu'))
        
        for l in self.hidden_dim[1::-1]:
            self.decoder.add(Dense(l, activation='relu'))
        
        self.decoder.add(Dense(self.input_dim, activation='linear'))
        
        # Define full VAE
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae')
        
        # Prepare optimizer
        optimizer = Adam(learning_rate = 1e-4)#, clipnorm=0.01)
        
        self.vae.compile(optimizer=optimizer, 
                         loss=self._loss, 
                         metrics=[self._mse_loss, self._kld_loss, self._mmd_loss], 
                         experimental_run_tf_function=False)
      
    def _loss(self, x_true, x_pred):
        """Returns loss function of the InfoVAE, uses multiple loss components
        to compute the total loss."""
        
        # Standard AE loss
        loss = self._mse_loss(x_true, x_pred)
        
        # Standard VAE loss: KL Divergenve
        KLD_loss = self._kld_loss(x_true, x_pred)
        
        # Additional InfoVAE loss: MMD Divergence
        MMD_loss = self._mmd_loss(x_true, x_pred)
        
        return loss + (1 - self.alpha) * KLD_loss + (self.lambd + self.alpha - 1) * MMD_loss
    
    def _mse_loss(self, x_true, x_pred):
        return K.sum(mse(x_true, x_pred))
    
    def _kld_loss(self, x_true, x_pred):
        return -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
    
    def _mmd_loss(self, x_true, x_pred):
        normal_samples = K.random_normal(shape=(K.shape(self.z_mean)[0], self.latent_dim))
        return self._compute_mmd(normal_samples, self.z)# * self.latent_dim * K.cast(K.shape(self.z)[0], 'float32')
        
    def _sampler(self, latent_layers):
        """Sample a distribution from the latent space variables, also
        known as the reparametrization trick.
        
        input:
            latent_layers:   a list with z_mean and z_log_var layers
        returns:
            the function: mu + sigma * epsilon, where epsilon
            is a random normal tensor"""

        # Unpack the latent layers
        z_mean, z_log_var = latent_layers
        
        # Extract size information for normal vector generation
        batch_size = K.shape(z_mean)[0]
        latent_dim = K.int_shape(z_mean)[1]
        
        # Normal vector with unit variance
        epsilon = K.random_normal(shape=(batch_size, latent_dim))
        
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
    def decode(self, *args, **kwargs):
        """Decode input to original dimension"""
        return self.decoder.predict(*args, **kwargs)
    
    def encode(self, *args, **kwargs):
        """Encode input to latent dimension"""
        return self.encoder.predict(*args, **kwargs)
    
    def fit(self, *args, **kwargs):
        """Hier weights opvangen?"""
        return self.vae.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.vae.predict(*args, **kwargs)

     
if __name__ == "__main__":

    # Load data
    data = np.memmap("SPECS_63839_SN5_5_8000.array.array", mode='r', dtype='float16', shape=(63839, 8000))
    specids = np.genfromtxt("SPECS_63839_SN5_5_8000.txt", unpack=True, dtype='str')
    
    # Split data
    original_dim = data.shape[1]
    x_train, x_test = train_test_split(data, test_size=0.1)

    # Network parameters
    hidden_dim = [1000, 500, 100]
    latent_dim = 6
    alpha = 0
    lambd = 10

    infoVAE = InfoVAE(original_dim,
                      hidden_dim,
                      latent_dim,
                      alpha=alpha,
                      lambd=lambd)
                      
    # Train network
    batch_size = 2048
    epochs = 1000
    es = EarlyStopping(patience=30, verbose=1)
    
    history = infoVAE.fit(x_train, x_train,
                      shuffle=True,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, x_test),
                      callbacks=[es],
                      use_multiprocessing=True)
                      
    # Save model parameters
    infoVAE.encoder.save_weights("encoder.h5")
    infoVAE.decoder.save_weights("decoder.h5")
    infoVAE.vae.save_weights("vae.h5")