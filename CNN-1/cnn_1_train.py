import numpy as np

from keras.layers import Conv2D,Input,Dense
from keras.layers import MaxPooling2D,Activation, AvgPool2D
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers import Flatten,Concatenate
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.callbacks import History, ModelCheckpoint,TensorBoard
from keras.utils import plot_model

import scipy.io as scio
from matplotlib import pyplot as plt





def shuffle_data(*params):
    '''
    :param params:Datasheets
    :return: Disrupted data
    '''
    params_num = len(params)
    length = len(params[0])
    index = np.random.permutation(length)
    result = []
    for i in range(params_num):
        result.append([])
    for i in index:
        for j in range(params_num):
            result[j].append(params[j][i])

    return result

if __name__ == "__main__":
    file_name = "./data.mat"
    data = scio.loadmat(file_name)
    np.random.seed(0)
    all_date = data["train_data_x"].transpose(3, 0, 1, 2)
    all_lable = data["train_data_y"].flatten()

    data = shuffle_data(all_date, all_lable)

    train_data_x = np.array(data[0][:540])
    train_data_y = np.array(data[1][:540])

    test_data_x = np.array(data[0][-1:])
    test_data_y = np.array(data[1][-1:])

    img_input = Input(shape=(16, 16, 3))
    x = Conv2D(6, (3, 3))(img_input)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AvgPool2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)


    adam = Adam()
    model = Model(inputs=img_input, outputs=output)
    model.compile(optimizer=adam, loss='mean_squared_error')

    plot_model(model,to_file='CNN-1.png')
    #modify
    checkpoint = ModelCheckpoint('CNN-1.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tb_callback = TensorBoard(log_dir='./CNN-1',histogram_freq=0,write_images=False, write_graph=True)
    history = model.fit(train_data_x, train_data_y, batch_size=int(train_data_x.shape[0]/10), epochs=10000, validation_split=1/9, callbacks=[checkpoint,tb_callback], verbose=2)
    epoch = history.epoch
    Loss = history.history
    plt.plot(epoch, Loss['val_loss'], epoch, Loss["loss"])
    plt.show()

    # model = load_model("./CNN-1.h5")
    # prediction = model.predict(test_data_x, batch_size=100)[:, 0]
    # aa = prediction - test_data_y
    # rate = len(np.where(np.abs(aa) < 0.5)[0]) / aa.shape[0]
    print('1')
