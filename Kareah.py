from tensorflow import keras
import glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
np.random.seed(1000)
'''
input: numpy array of shape(batch_size,input_dim_0,input_dim_1,1)
Y: keras tensor of shape(batch_size,8,2,2)
'''

def convert_str_to_float(list_str):
    res = []
    for str in list_str:
        if str == "":
            continue
        res.append(float(str))
    return res


def min_max_normal():
    return


def read_mat(file_name):
    res = []
    list_of_lists = []
    with open(file_name) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            # in alternative, if you need to use the file content as numbers
            # inner_list = [int(elt.strip()) for elt in line.split(',')]
            list_of_lists.append(inner_list)

    for item in list_of_lists:
        res.append(convert_str_to_float(item))
    return np.asarray(res).reshape(300,120,1)


def prepare_train_data(test_dir):
    classes = []
    labels = []
    for file in glob.glob(test_dir + "\*.txt"):
        classes.append(read_mat(file))
    for file in glob.glob(test_dir + "\*.dat"):
        labels.append(read_mat(file))
    # convert to ndarray
    # classes, labels = np.asarray(classes, dtype=None, order=None),np.asarray(labels, dtype=None, order=None)
    return np.asarray(classes), np.asarray(labels)

def list_of_res(c_mats,l_mats):
    res = []
    for mat1,mat2 in zip(c_mats,l_mats):
        temp = np.linalg.pinv(mat1)
        res.append(np.matmul(temp, mat2))
    return res



def main():
    # prepare data
    train_X, train_Y = prepare_train_data(
        r"C:\Users\leah2\OneDrive\שולחן העבודה\hw\Lab Projects\muchine learnning"
        r"\Var 1\train")
    test_X, test_Y = prepare_train_data(
        r"C:\Users\leah2\OneDrive\שולחן העבודה\hw\Lab Projects\muchine learnning"
        r"\Var 1\test")
    print('Training data shape : ', train_X.shape, train_Y.shape)

    print('Testing data shape : ', test_X.shape, test_Y.shape)
    # Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(300, 120, 1), kernel_size=(11, 11), strides=(4, 4), padding="valid"))
    model.add(Activation("relu"))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding="valid"))
    model.add(Activation("relu"))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="valid"))
    model.add(Activation("relu"))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding="valid"))
    model.add(Activation("relu"))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="valid"))
    model.add(Activation("relu"))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
    model.add(Activation("relu"))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation("relu"))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation("relu"))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(17))
    model.add(Activation("softmax"))

    model.summary()

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics = ["accuracy"])


    #model.add(Conv2D(1098752, kernel_size=3, activation="softmax"))
    #model.summary()

    #model.compile(optimizer='adam', loss='mean_squared_error', metrics=['Accuracy'])

    #model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=3000000)
    #model.save(r'C:\Users\leah2\OneDrive\שולחן העבודה\hw\Lab Projects\muchine learnning')


if __name__ == '__main__':
    main()