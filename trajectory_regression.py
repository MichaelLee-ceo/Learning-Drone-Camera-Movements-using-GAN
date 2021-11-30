import numpy as np
import matplotlib.pyplot as plt
from read_data import read_coordinates
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import torch


def PolynomialRegression(num):
    poly_model = make_pipeline(PolynomialFeatures(int(num)), LinearRegression())
    return poly_model


degree = 5
train_data = read_coordinates()[:100].numpy()

for directory in range(100):
    # if not os.path.exists('pic/'+str(dire)+'/'):
    #     os.makedirs('pic/'+str(dire)+'/')

    train_dat = train_data[directory]
        # print(train_data[0])
        # print(train_dat.shape)
    xy_train=train_dat[0:38,0:2]
    x_data = train_dat[0:38,0].astype(int)
    y_data = train_dat[0:38,1].astype(int)
    z_train=train_dat[:,2]
    xy_train = xy_train.astype(int)
    z_train = z_train.astype(int)

    # 對Y做FIT
    model = PolynomialRegression(degree)
    model.fit(x_data[:,np.newaxis],z_train)
    zfit = model.predict(x_data[:, np.newaxis])

    # 對Z做FIT
    model = PolynomialRegression(degree)
    model.fit(x_data[:,np.newaxis],y_data)
    yfit = model.predict(x_data[:, np.newaxis])

    # 畫圖
    fig = plt.figure()
    ax=fig.add_subplot(projection='3d')

    # 放原本的點
    ax.scatter(x_data,y_data,z_train,c='red')
    # ax.scatter(x_data,y_data,zfit,c='blue')

    # 放FIT後的plot
    ax.plot(x_data,yfit,zfit)
    fig.savefig('pic/'+str(directory)+'.png', dpi=200)

    # 存.pt檔去train
    sumAr = np.vstack([x_data,yfit,zfit])
    torch.save(torch.tensor(np.transpose(sumAr), dtype=torch.float32), './coordinates_camera_pos2/shot_' + str(dire) + '.pt')
