import numpy as np
from sklearn.linear_model import LinearRegression
from dist_ml import distance
from sklearn.metrics import mean_absolute_error


def cal_angle_x(px , depth):
    # data
    #                       1         2          3          10         20        25.3
    px_train = np.array([[37.795], [75.591], [113.386], [377.953], [755.906], [956.22]])
    ang_train = distance(depth)

    # Training 
    model = LinearRegression()
    model.fit(px_train, ang_train)
    
    #prediction
    z = np.array([[px]])
    c_angle = model.predict(z)
    c_angle = c_angle.item()
    c_angle = round(c_angle,1)
    #print(c_angle)
    return c_angle

#cal_angle_x(377.953,68)