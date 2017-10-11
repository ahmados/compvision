# %load /Users/ahmad/Desktop/02-seam-carve/seam_carve.py
import numpy as np
from numpy import zeros
 
def gradi(arr):
    gradx = zeros(arr.shape)
    grady = zeros(arr.shape)
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            if (i == 0):
                gradx[i][j] = arr[i + 1][j] - arr[i][j]
            elif (i == (arr.shape[0] - 1)):
                gradx[i][j] = arr[i][j] - arr[i - 1][j]
            else:
                gradx[i][j] = arr[i + 1][j] - arr[i - 1][j]
            if (j == 0):
                grady[i][j] = arr[i][j + 1] - arr[i][j]
            elif (j == (arr.shape[1] - 1)):
                grady[i][j] = arr[i][j] - arr[i][j - 1]
            else:
                grady[i][j] = arr[i][j + 1] - arr[i][j - 1]
    return (gradx ** 2 + grady ** 2) ** 0.5
 
def seam_carve(img, mode, mask=None):
    # dummy implementation — delete rightmost column of image
    resized_img = img[:, :-1, :]
    gray = 0.299 * img[..., 0]
    gray[:] += 0.587 * img[..., 1]
    gray[:] += 0.114 * img[..., 2]
    import numpy as np
    arr = gray
#         grad = gradi(arr)
    gradx, grady = np.gradient(arr)
    gradx *= 2
    grady *= 2
    for i in range(gradx.shape[1]):
        gradx[0, i] = (gradx[0, i] / 2)
        gradx[gradx.shape[0] - 1, i] = (gradx[gradx.shape[0] - 1, i] / 2)
    for i in range(gradx.shape[0]):
        grady[i, 0] = (grady[i, 0] / 2)
        grady[i, gradx.shape[1] - 1] = (grady[i, gradx.shape[1] - 1] / 2)
    grad = np.sqrt(np.square(gradx) + np.square(grady))
    seamMatrix = zeros((grad.shape[0], grad.shape[1], 2))
    if mask is None:
        resized_mask = None
    else:
        masker = 256 * seamMatrix.shape[0] * seamMatrix.shape[1]
        resized_mask = mask[:, :-1]
        grad += masker * mask
    carve_mask = zeros((img.shape[0], img.shape[1]))
    if ((mode == 'horizontal shrink') or (mode == 'horizontal expand')):
        for k in range(0, grad.shape[1]):
            seamMatrix[0][k][0] = grad[0][k]
            seamMatrix[0][k][1] = 0
        for i in range(1, grad.shape[0]):
            for j in range(0, grad.shape[1]):
                if (j == 0):
                    if (seamMatrix[i - 1, j, 0] <= seamMatrix[i - 1, j + 1, 0]):
                        seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i - 1, j, 0]
                        seamMatrix[i, j, 1] = j
                    else:
                        seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i - 1, j + 1, 0]
                        seamMatrix[i, j, 1] = j + 1
                elif (j == (grad.shape[1] - 1)):
                    if (seamMatrix[i - 1, j - 1, 0] <= seamMatrix[i - 1, j, 0]):
                        seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i - 1, j - 1, 0]
                        seamMatrix[i, j, 1] = j - 1
                    else:
                        seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i - 1, j, 0]
                        seamMatrix[i, j, 1] = j
                else:
                    if (seamMatrix[i - 1, j - 1, 0] <= seamMatrix[i - 1, j, 0]):
                        if (seamMatrix[i - 1, j - 1, 0] <= seamMatrix[i - 1, j + 1, 0]):
                            seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i - 1, j - 1, 0]
                            seamMatrix[i, j, 1] = j - 1
                        else:
                            seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i - 1, j + 1, 0]
                            seamMatrix[i, j, 1] = j + 1
                    else:
                        if (seamMatrix[i - 1, j, 0] <= seamMatrix[i - 1, j + 1, 0]):
                            seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i - 1, j, 0]
                            seamMatrix[i, j, 1] = j
                        else:
                            seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i - 1, j + 1, 0]
                            seamMatrix[i, j, 1] = j + 1
        indexLast = np.argmin(seamMatrix[(grad.shape[0] - 1), :, 0])
        indexLast = int(round(indexLast))
        for t in range(0, grad.shape[0]):
            carve_mask[grad.shape[0] - 1 - t][indexLast] = 1
            indexLast = seamMatrix[grad.shape[0] - 1 - t][indexLast][1]
            indexLast = int(round(indexLast))
    else:
        for k in range(0, grad.shape[0]):
            seamMatrix[k][0][0] = grad[k][0]
            seamMatrix[k][0][1] = 0
        for j in range(1, grad.shape[1]):
            for i in range(0, grad.shape[0]):
                if (i == 0):
                    if (seamMatrix[i , j - 1, 0] <= seamMatrix[i + 1, j - 1, 0]):
                        seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i, j - 1, 0]
                        seamMatrix[i, j, 1] = i
                    else:
                        seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i + 1, j - 1, 0]
                        seamMatrix[i, j, 1] = i + 1
                elif (i == (grad.shape[0] - 1)):
                    if (seamMatrix[i - 1, j - 1, 0] <= seamMatrix[i, j - 1, 0]):
                        seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i - 1, j - 1, 0]
                        seamMatrix[i, j, 1] = i - 1
                    else:
                        seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i, j - 1, 0]
                        seamMatrix[i, j, 1] = i
                else:
                    if (seamMatrix[i - 1, j - 1, 0] <= seamMatrix[i, j - 1, 0]):
                        if (seamMatrix[i - 1, j - 1, 0] <= seamMatrix[i + 1, j - 1, 0]):
                            seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i - 1, j - 1, 0]
                            seamMatrix[i, j, 1] = i - 1
                        else:
                            seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i + 1, j - 1, 0]
                            seamMatrix[i, j, 1] = i + 1
                    else:
                        if (seamMatrix[i , j - 1, 0] <= seamMatrix[i + 1, j - 1, 0]):
                            seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i, j - 1, 0]
                            seamMatrix[i, j, 1] = i
                        else:
                            seamMatrix[i, j, 0] = grad[i, j] + seamMatrix[i + 1, j - 1, 0]
                            seamMatrix[i, j, 1] = i + 1
#                         tempArr = (seamMatrix[i - 1][j - 1][0], seamMatrix[i][j - 1][0], seamMatrix[i + 1][j - 1][0])
#                         seamMatrix[i][j][0] = min(tempArr) + grad[i][j]
#                         seamMatrix[i][j][1] = (tempArr.index(min(tempArr))) + i - 1
        indexLast = np.argmin(seamMatrix[:, grad.shape[1] - 1, 0])
        indexLast = int(round(indexLast))
        for t in range(0, grad.shape[1]):
            carve_mask[indexLast][grad.shape[1] - 1 - t] = 1
            indexLast = seamMatrix[indexLast][grad.shape[1] - 1 - t][1]
            indexLast = int(round(indexLast))
    return (resized_img, resized_mask, carve_mask)