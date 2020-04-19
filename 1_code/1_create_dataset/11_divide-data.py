import numpy as np
from sklearn.model_selection import train_test_split

X_path = '/Users/amandeeprathee/work/fake-or-not/00_source_data/numpy-binaries/ps_battles_ela_X.npy'

Y_path = '/Users/amandeeprathee/work/fake-or-not/00_source_data/numpy-binaries/ps_battles_ela_Y.npy'

# load files into numpy object
X = np.load(X_path)
X.shape
Y = np.load(Y_path)
Y.shape

# divide data
test_size = int(X.shape[0]*0.15)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=1, shuffle=False)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=test_size, random_state=1, shuffle=False)

# check shapes of each array
X_train.shape
Y_train.shape

X_validation.shape
Y_validation.shape

X_test.shape
Y_test.shape

# save all train, validation and test data to .npy file
np.save('/Users/amandeeprathee/work/fake-or-not/00_source_data/numpy-binaries/X_train.npy', X_train)
np.save('/Users/amandeeprathee/work/fake-or-not/00_source_data/numpy-binaries/Y_train.npy', Y_train)

np.save('/Users/amandeeprathee/work/fake-or-not/00_source_data/numpy-binaries/X_validation.npy', X_validation)
np.save('/Users/amandeeprathee/work/fake-or-not/00_source_data/numpy-binaries/Y_validation.npy', Y_validation)

np.save('/Users/amandeeprathee/work/fake-or-not/00_source_data/numpy-binaries/X_test.npy', X_test)
np.save('/Users/amandeeprathee/work/fake-or-not/00_source_data/numpy-binaries/Y_test.npy', Y_test)
