import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Concatenate, Dropout
from keras.optimizers import Adam
from tqdm import tqdm
from loadmain import load_configs
from datetime import datetime
import os
import shutil


def hot_encode_num(n):
    return [0 if x != n else 1 for x in range(10)]


def plot_loss(gan_loss, disc_loss, init: bool = False):
    epochs = list(range(1, len(gan_loss) + 1))

    plt.clf()
    if init:
        plt.title(f"Training epoch: {len(epochs)+1}")
    else:
        plt.title(
            f'Training epoch: {len(epochs) + 1}, DL(r): {disc_loss[-1]:.4f}, GL(b): {gan_loss[-1]:.4f}')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylim(ymin=0,)

    plt.plot(gan_loss, "b")
    plt.plot(disc_loss, "r")

    plt.show(block=False)
    plt.pause(1)


if __name__ == "__main__":

    print("\n\n\n")

    # Load MNIST dataset, but labels as X and image as y
    (train_y, train_X), (_, _) = mnist.load_data()

    # Data preprocessing
    train_y = (train_y.astype("float32") - 127.5) / 127.5
    train_y = train_y.reshape(train_y.shape[0], 28 * 28)
    train_X = np.apply_along_axis(
        hot_encode_num, 1, train_X.reshape(train_X.shape[0], 1))

    # Load configs
    configs = load_configs()
    print(configs)

    # Options
    noise_amount = configs["noise_amount"]
    inp_classes = 10
    epochs = configs["epochs"]
    batch_size = configs["batch_size"]
    save_interval = configs["save_interval"]

    # GAN hyperparameters
    adam = Adam(
        learning_rate=configs["learning_rate"], beta_1=configs["beta_1"])
    activation = LeakyReLU(0.3)

    # Generator Model
    gen_model = Sequential([
        Input(shape=(noise_amount + inp_classes,)),
        Dense(256, activation=activation),
        Dropout(0.2),
        BatchNormalization(),
        Dense(512, activation=activation),
        Dropout(0.2),
        BatchNormalization(),
        Dense(1024, activation=activation),
        Dense(784, activation='sigmoid')
    ])
    noise = Input(shape=(noise_amount))
    label = Input(shape=(inp_classes))
    X = Concatenate(axis=-1)([noise, label])
    img = gen_model(X)
    generator = Model([noise, label], img)
    generator.compile(optimizer=adam, loss="mse")

    # Discriminator Model
    disc_model = Sequential([
        Input(shape=(784 + inp_classes),),
        Dense(1024, activation=activation),
        Dropout(0.2),
        Dense(512, activation=activation),
        Dropout(0.2),
        Dense(256, activation=activation),
        Dense(1, activation="sigmoid"),
    ])
    img = Input(shape=(784,))
    label = Input(shape=(inp_classes,))
    X = Concatenate(axis=-1)([img, label])
    pred = disc_model(X)
    discriminator = Model([img, label], pred)
    discriminator.compile(optimizer=adam, loss="binary_crossentropy")

    # GAN Model, essentially: discriminator(generator(input) + label)
    noise = Input(shape=(noise_amount))
    label = Input(shape=(inp_classes,))
    img = generator([noise, label])
    pred = discriminator([img, label])

    gan = Model([noise, label], pred)
    gan.compile(loss="binary_crossentropy", optimizer=adam)
    gan.summary()

    gan_loss_hist, disc_loss_hist = [], []

    # Create save directory
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%m-%d-%y_%H.%M")
    folder_name = f"GAN_{formatted_datetime}"
    folder_path = "./models/" + folder_name
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path + "/images", exist_ok=True)
    shutil.copy("settings.cfg", folder_path)
    print("Directory created at " + folder_path)

    # Training
    for e in range(1, epochs + 1):
        noise = np.random.normal(0, 1, size=(train_X.shape[0], noise_amount))
        generator.fit(x=[noise, train_X], y=train_y,
                      batch_size=batch_size, verbose=1)

        generated_imgs = generator.predict([noise, train_X])

        X_imgs = np.concatenate([generated_imgs, train_y], axis=0)
        X_labels = np.concatenate([train_X, train_X], axis=0)
        y_classification = np.concatenate(
            [np.zeros(shape=(train_X.shape[0])), np.ones(shape=(train_X.shape[0]))], axis=0)

        disc_loss = discriminator.fit(
            x=[X_imgs, X_labels], y=y_classification, batch_size=batch_size, verbose=1).history['loss'][0]

        new_noise = np.random.normal(
            0, 1, size=(train_X.shape[0], noise_amount))

        y_classification = np.ones(shape=(train_X.shape[0],))
        gan_loss = gan.fit([new_noise, train_X],
                           y_classification, batch_size=batch_size, verbose=1).history['loss'][0]

        print(
            f"Epoch {e}/{epochs}, Discriminator Loss: {disc_loss:.4f}, Generator Loss: {gan_loss:.4f}")
        gan_loss_hist.append(round(gan_loss, 4))
        disc_loss_hist.append(round(disc_loss, 4))

        # plot_loss(gan_loss_hist, disc_loss_hist)
        if (e % save_interval == 0) or (e == epochs + 1):
            print(f"epoch {e} benchmark")
            plotter = ModelVisualMetrics(generator)
            plot = plotter.preview_all(return_only=configs["skip_preview"])
            plot.savefig(f"{folder_path}/images/epoch_{e}")
            generator.save(f"{folder_path}/GenModel_{e}")
            print(f"model saved as: {folder_path}/GenModel_{e}")

    # Training complete
    print("training complete")
