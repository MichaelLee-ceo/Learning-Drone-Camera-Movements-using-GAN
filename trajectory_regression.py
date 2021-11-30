import numpy as np
import matplotlib.pyplot as plt
from read_data import read_coordinates
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import torch
import os


def PolynomialRegression(num):
    poly_model = make_pipeline(PolynomialFeatures(int(num)), LinearRegression())
    return poly_model

if not os.path.exists('regression_result/'):
    os.makedirs('regression_result/')
if not os.path.exists('coordinates_camera_pos_regression/'):
    os.makedirs('coordinates_camera_pos_regression/')

degree = 5
train_data = read_coordinates()[:100].numpy()
print('Train data shape:', train_data.shape)

for directory in range(100):
    train_dat = train_data[directory]

    x_data = train_dat[0:38,0].astype(int)
    y_data = train_dat[0:38,1].astype(int)
    z_data = train_dat[:, 2].astype(int)

    # fit on z
    model = PolynomialRegression(degree)
    model.fit(x_data[:,np.newaxis],z_data)
    zfit = model.predict(x_data[:, np.newaxis])

    # fit on y
    model = PolynomialRegression(degree)
    model.fit(x_data[:,np.newaxis],y_data)
    yfit = model.predict(x_data[:, np.newaxis])

    # 畫圖
    fig = plt.figure()
    ax=fig.add_subplot(projection='3d')

    # 放原本的點
    ax.scatter(x_data,z_data,y_data,c='red')
    ax.scatter(x_data, zfit, yfit, c='blue')

    # 放FIT後的plot
    ax.plot(x_data,zfit,yfit)

    fig.savefig('regression_result/'+str(directory)+'.png', dpi=200)
    plt.close()

    # save .pt file for training data
    sumAr = np.vstack([x_data,yfit,zfit])
    torch.save(torch.tensor(np.transpose(sumAr), dtype=torch.float32), './coordinates_camera_pos_regression/shot_' + str(directory) + '.pt')
    print("[INFO]: torch save " + './coordinates_camera_pos_regression/shot_' + str(directory) + '.pt')
