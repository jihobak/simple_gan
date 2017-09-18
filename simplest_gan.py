import os
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Dropout, LeakyReLU
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data  # load mnist
from tensorflow.contrib.learn.python.learn.datasets import mnist


mnist.SOURCE_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
data = input_data.read_data_sets('fmnist_data') # 'fmnist_data' is directory name for saving fmnist data.


FMNIST_DIM = 28*28*1
LATENT_DIM = 100
batch_size = 20
steps = 10000


def save_images(images, save_path, filename):
    """
    save multiple images generated from generator(forger)
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    how_many = images.shape[0]

    fig = plt.figure(figsize=(5, 5))  # Notice the equal aspect ratio
    ax = [fig.add_subplot(5, round(batch_size/5), i + 1) for i in range(how_many)]

    for i, a in enumerate(ax):
        a.axis('off')
        a.set_aspect('equal')
        a.imshow(images[i].reshape(28, 28), cmap='gray')

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()
    plt.savefig(os.path.join(save_path, filename))


def generator(input_z):
    z = Dense(128, activation='relu')(input_z)
    #z = LeakyReLU()(z)
    fake = Dense(FMNIST_DIM, activation='sigmoid', name='generator_output')(z)

    generator = Model(input_z, fake)
    # generator.summary()

    return generator


def discriminator(input_data):
    x = Dense(128, activation='relu')(input_data)
    #x = LeakyReLU()(x)
    #x = Dropout(rate=0.5)(x)
    probability = Dense(1, activation='sigmoid', name='discriminator_output')(x)

    discriminator = Model(input_data, probability)
    # discriminator.summary()

    return discriminator


# Make 'input' for out GAN model
imgs = Input(shape=(FMNIST_DIM,))
latent_space = Input(shape=(LATENT_DIM,))
an_input = Input(shape=(LATENT_DIM,))


# Make g & d
forger = generator(latent_space)
expert = discriminator(imgs)


expert_optimizer = Adam()
expert.compile(optimizer=expert_optimizer, loss='binary_crossentropy')


expert.trainable  = False # import step for training GAN model.
an_input = Input(shape=(LATENT_DIM,))
an_output = expert(forger(an_input))

adversarial_network = Model(an_input, an_output)
forger_optimizer = Adam()

adversarial_network.compile(optimizer=forger_optimizer, loss='binary_crossentropy')


save_path = 'result'

for step in range(steps):

    # 1. Getting real images(fminst)
    real_images, _ = data.train.next_batch(batch_size)

    # 2. Make data set for training discriminator(expert)
    random_latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
    generated_images = forger.predict(random_latent_vectors)

    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([np.zeros((batch_size, 1)),
                             np.ones((batch_size, 1))])

    # 3-1. Train the discriminator(expert)
    d_loss = expert.train_on_batch(combined_images, labels)

    # Make data set for training generator(forger)
    # sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
    labels = np.ones((batch_size, 1))

    # 3-2. Train the generator (the discriminator weights are frozen)
    a_loss = adversarial_network.train_on_batch(random_latent_vectors, labels)

    if (step % 500 == 0):
        # Save model weights
        adversarial_network.save_weights('gan.h5')

        # Print metrics
        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        # Save generated images
        save_images(generated_images, save_path, 'generated_fashion' + str(step) + '.png')