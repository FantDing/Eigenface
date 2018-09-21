from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os

# 人脸数据下载地址：http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip
root = "./att_faces"


def make_input(faces_path):
    img = io.imread(faces_path[0])
    matrix = img.flatten()[:, np.newaxis]  # (10304,)与（10304,）不能叠加，因为超出维度了。必须再增加一维
    for i in range(len(faces_path) - 1):
        img = io.imread(faces_path[i + 1])
        img = img.flatten()[:, np.newaxis]
        matrix = np.concatenate((matrix, img), axis=1)
    return matrix


def showEigFace(Eig, row, col):
    plt.figure()
    for index in range(row * col):
        plt.subplot(row, col, index + 1)
        plt.imshow(Eig[:, index].reshape(origin.shape), cmap="gray")
    plt.show()


def PCA(X):
    # 减去均值
    x_avg = np.mean(X, axis=1, keepdims=True)
    X = X - x_avg
    # 协方差矩阵
    # https://zh.wikipedia.org/wiki/%E7%89%B9%E5%BE%81%E8%84%B8 加快计算
    C = np.dot(X.T, X) / X.shape[0]
    # 求特征值，特征向量
    eig_value, eig_vec = np.linalg.eigh(C)
    eig_vec = X.dot(eig_vec)
    norm_2 = np.linalg.norm(eig_vec, axis=0)
    eig_vec = eig_vec / norm_2
    # 将特征值、特征向量进行倒序排列(大->小)
    eig_value = eig_value[::-1]
    eig_vec = eig_vec[:, ::-1]
    return eig_value, eig_vec


if __name__ == "__main__":
    faces_cls = os.listdir(root)
    faces_img_path = [os.path.join(root, cls, '1.pgm') for cls in faces_cls]
    origin = io.imread(faces_img_path[0])
    print(origin.shape[0] * origin.shape[1])
    X = make_input(faces_img_path)
    eig_value, eig_vec = PCA(X.astype(np.float))  # uint8->float
    print(eig_value[:4])
    showEigFace(eig_vec, 2, 3)
