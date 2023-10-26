import keras.datasets.mnist
import numpy as np
from keras import Sequential, losses
from keras.layers import *
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class DigitClassifier:

    def __init__(self, inp_cap=None, debug=False, save=False):
        (self.X_train, self.y_train), (self.X_test,
                                       self.y_test) = keras.datasets.mnist.load_data()[:inp_cap]
        self.inp_cap = inp_cap if inp_cap is not None else len(self.X_train)
        self.X_train = self.X_train.astype('float')
        self.X_test = self.X_test.astype('float')
        print(self.X_train.shape, self.y_train.shape,
              self.X_test.shape, self.y_test.shape)

        self.X_train[self.X_train < 200] = 0  # filter out weaker brightnesses
        self.X_test[self.X_test < 200] = 0  # filter out weaker brightnesses
        # strengthen stronger brightnesses
        self.X_train[self.X_train >= 200] = 255
        # strengthen stronger brightnesses
        self.X_test[self.X_test >= 200] = 255

        conv_activation = 'relu'
        self.model = Sequential([
            Reshape((28, 28, 1)),

            Conv2D(64, (3, 3), padding='same', activation=conv_activation),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.3),

            Conv2D(64, (3, 3), padding='same', activation=conv_activation),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.3),

            Conv2D(64, (3, 3), padding='same', activation=conv_activation),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.3),

            Flatten(),
            Dense(256, activation='relu'),  # perform classification
            Dense(10, activation='sigmoid'),
        ])
        self.model.compile(loss=keras.losses.SparseCategoricalCrossentropy(
        ), optimizer="adam", metrics=["accuracy"], )

        self.history = None

        self.debug = debug
        self.save = save

        # self.train_model()

    def train_model(self):
        print('Training Model')
        self.history = self.model.fit(
            self.X_train, self.y_train, epochs=50, validation_data=(self.X_test, self.y_test))
        print("Training Completed")
        self.model.summary()
        if self.save:
            inp = input("Save Model? Y/N\n")
            if inp == "Y":
                self.model.save('DigitModel')
                print('Model Saved')
            else:
                print("Model Not Saved")
        return self.history

    def predict(self, img: np.array):
        if len(img.shape) < 3:  # if singular image turn into 3d array
            img = img.reshape(1, *img.shape)
        assert img.shape == (1, 28, 28)
        hot_encoded_result = self.model.predict(img)
        return np.argmax(hot_encoded_result)

    def get_random_index(self):
        return np.random.randint(0, len(self.X_test))

    def plot_one_image(self, img=None):  # grabs a random image from X_test
        if img is None:
            img = self.X_test[self.get_random_index()]
        assert img.shape == (28, 28)
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()

    def plot_acc(self, ax=None, xlabel='Epoch #'):
        if self.history is not None:
            if hasattr(self.history, 'history_'):
                history = self.history.history_
            else:
                history = self.history.history
            history.update(
                {'epoch': list(range(len(history['val_accuracy'])))})
            history = pd.DataFrame.from_dict(history)

            best_epoch = history.sort_values(
                by='val_accuracy', ascending=False).iloc[0]['epoch']

            if not ax:
                f, ax = plt.subplots(1, 1)
            sns.lineplot(x='epoch', y='val_accuracy',
                         data=history, label='Validation', ax=ax)
            sns.lineplot(x='epoch', y='accuracy',
                         data=history, label='Training', ax=ax)
            ax.axhline(0.1, linestyle='--', color='red', label='Chance')
            ax.axvline(x=best_epoch, linestyle='--',
                       color='green', label='Best Epoch')
            ax.legend(loc=7)
            ax.set_ylim([0.4, 1])

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Accuracy (Fraction)')

            plt.show()


if __name__ == '__main__':
    knn = DigitClassifier(save=True)  # train a new classifier
    knn.train_model()

    # Extra
    knn.plot_acc()
    index = knn.get_random_index()
    img = knn.X_test[index]
    knn.plot_one_image(img)
    print()
    print(f"predicted: {knn.predict(img)}")
