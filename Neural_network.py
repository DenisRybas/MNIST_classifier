import tensorflow as tf


class NeuralNetworkModel:
    model = None

    def __init__(self):
        self.model = tf.keras.models.load_model('num_reader.model')
        # self.model = self.create_model()
        # self.save_model()

    def create_model(self):
        dataset_mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written numbers 0-9
        (x_train, y_train), (x_test, y_test) = dataset_mnist.load_data()
        x_train = tf.keras.utils.normalize(x_train, axis=1)
        x_test = tf.keras.utils.normalize(x_test, axis=1)

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10)
        return model

    def save_model(self):
        self.model.save('num_reader.model')
