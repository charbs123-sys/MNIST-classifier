from prac import *
import pandas as pd

images, labels = convert_to_binary('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')

df = pd.DataFrame(images[0])
vector_features = pd.Series([0] * (28 * 28))

X_train = images
y_train = labels

rng = np.random.RandomState(42)

weight = rng.standard_normal(size = (28 * 28))

train_1 = np.ravel(X_train[0])
print(train_1)
