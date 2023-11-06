from tensorflow import keras
from sklearn.model_selection import train_test_split


# download fashion mnist data
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28 * 28)

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

dense = keras.layers.Dense(10, activation="softmax", input_shape=(784,))
model = keras.Sequential(dense)
model.compile(loss="sparse_categorical_crossentropy", metrics="accuracy")

model.fit(train_scaled, train_target, epochs=5)
