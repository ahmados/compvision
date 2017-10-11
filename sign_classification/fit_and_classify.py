import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from skimage.filters import sobel_h, sobel_v
from skimage.transform import rescale, resize, downscale_local_mean

def extract_hog(img):
    img = resize(img, (48, 48))
    img_resized = img
    edgesHRed = sobel_h(img[:, :, 0])
    edgesHGreen = sobel_h(img[:, :, 1])
    edgesHBlue = sobel_h(img[:, :, 2])
    edgesVRed = sobel_v(img[:, :, 0])
    edgesVGreen = sobel_v(img[:, :, 1])
    edgesVBlue = sobel_v(img[:, :, 2])
    edgesRed = np.sqrt(np.square(edgesHRed) + np.square(edgesVRed))
    edgesGreen = np.sqrt(np.square(edgesHGreen) + np.square(edgesVGreen))
    edgesBlue = np.sqrt(np.square(edgesHBlue) + np.square(edgesVBlue))
    grad = np.zeros(img.shape[:-1])
    phi = np.zeros(grad.shape)
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            grad[i][j] = np.amax([edgesRed[i][j], edgesGreen[i][j], edgesBlue[i][j]])
            ind = np.argmax([edgesRed[i][j], edgesGreen[i][j], edgesBlue[i][j]])
            if (ind == 0):
                phi[i][j] = np.arctan2(edgesHRed[i][j], edgesVRed[i][j])
            elif (ind == 1):
                phi[i][j] = np.arctan2(edgesHGreen[i][j], edgesVGreen[i][j])
            else:
                phi[i][j] = np.arctan2(edgesHBlue[i][j], edgesVBlue[i][j])
    phi = np.absolute(phi)

#     leastSide = min(img.shape[0], img.shape[1])

#     if (leastSide < 40):
#         img_resized = resize(img, (32, 32))
#     elif ((leastSide >= 40) and (leastSide < 48)):
#         img_resized = resize(img, (48, 48))
#     elif ((leastSide >= 48) and (leastSide < 56)):
#         img_resized = resize(img, (56, 56))
#     else:
#         img_resized = resize(img, (64, 64))

    x = img_resized.shape[0] // 8
    vector = np.zeros((x ** 2, 9))
    resVector = np.zeros(0)

    for i in range(0, x):
        for j in range(0, x):
            curPic = img_resized[i : (i + 8), j : (j + 8)]
            for k in range(8):
                for t in range(8):
                    sector = int((phi[8 * i + k, 8 * j + t] / (np.pi / 8)))
                    weight = ((phi[8 * i + k, 8 * j + t] - (np.pi / 8) * sector) / (np.pi / 8))
                    vector[i * x + j, sector] += weight * grad[i][j]
                    if (sector != 8):
                        vector[i * x + j, sector + 1] += (1 - weight) * grad[8 * i + k][8 * j + t]
                    else:
                        vector[i * x + j, 0] += (1 - weight) * grad[8 * i + k][8 * j + t]
    for i in range (x - 1):
        for j in range(x - 1):
            tempArr1 = np.append(vector[i * x + j], vector[i * x + j + 1])
            tempArr2 = np.append(vector[i * x + x + j], vector[i * x + x + j + 1])
            tempArr = np.append(tempArr1, tempArr2)
            s = np.square(tempArr)
            sumnorm = np.sum(s)
            tempArr = tempArr / (np.sqrt(np.sum(s) + 0.0002))
            resVector = np.append(resVector, tempArr)
    return resVector


#comment

def fit_and_classify(train_features, train_labels, test_features):
    from sklearn import svm
    clf = svm.LinearSVC()
    clf.fit(train_features, train_labels)
    data = clf.predict(test_features)
    return data