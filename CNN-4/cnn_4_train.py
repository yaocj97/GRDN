import numpy as np

from keras.layers import Conv2D,Input,Dense
from keras.layers import MaxPooling2D,Activation, AvgPool2D
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
    file_name = "./cnn4_data.mat"
    data = scio.loadmat(file_name)
    np.random.seed(0)
    all_date = data["train_data_x"].transpose(3, 0, 1, 2)
    all_lable = data["train_data_y"].flatten()

    data = shuffle_data(all_date, all_lable)

    train_data_x = np.array(data[0][:-180])
    train_data_y = np.array(data[1][:-180])
    # train_data_x2 = train_data_x.copy() + np.random.rand(train_data_x.shape[0], train_data_x.shape[1], train_data_x.shape[2], train_data_x.shape[3])*0.02 - 0.01
    # train_data_y2 = train_data_y.copy()
    # train_date_x = np.row_stack((train_data_x, train_data_x2))
    # train_date_y = np.row_stack((train_data_y, train_data_y2))



    test_data_x = np.array(data[0][-180:])
    test_data_y = np.array(data[1][-180:])

    img_input = Input(shape=(16, 32, 3))
    x = Conv2D(6, (5, 5))(img_input)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AvgPool2D(pool_size=(2, 2))(x)
    x = Conv2D(12, (3, 3))(x)
    x = Activation('relu')(x)
    x = AvgPool2D(pool_size=(1, 2))(x)
    x = Flatten()(x)
    x = Dense(24)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)


    adam = Adam()
    # adam = SGD()
    model = Model(inputs=img_input, outputs=output)
    model.compile(optimizer=adam, loss='mean_squared_error')

    plot_model(model,to_file='CNN-4.png')
    #modify
    checkpoint = ModelCheckpoint('CNN-4.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tb_callback = TensorBoard(log_dir='./CNN-4',histogram_freq=0,write_images=False, write_graph=True)
    history = model.fit(train_data_x, train_data_y, batch_size=int(train_data_x.shape[0]/10), epochs=1000, validation_split=1/9, callbacks=[checkpoint,tb_callback], verbose=2)
    epoch = history.epoch
    Loss = history.history
    plt.plot(epoch, Loss['val_loss'], epoch, Loss["loss"])
    plt.show()

    # model = load_model("./CNN-4.h5")
    # prediction = model.predict(test_data_x, batch_size=100)[:, 0]
    # aa = prediction - test_data_y
    # rate = len(np.where(np.abs(aa) < 0.5)[0]) / aa.shape[0]
    print('1')