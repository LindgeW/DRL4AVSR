import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
import dcor


def center_distance_matrix(D):
    """
    中心化距离矩阵
    :param D: 距离矩阵
    :return: 中心化后的距离矩阵
    """
    n = D.shape[0]
    H = torch.eye(n, device=D.device) - torch.ones((n, n), device=D.device) / n
    return torch.mm(torch.mm(H, D), H)


def distance_covariance(X, Y):
    """
    计算距离协方差
    :param X: 第一个随机变量的样本
    :param Y: 第二个随机变量的样本
    :return: 距离协方差
    """
    n = X.shape[0]
    # 计算距离矩阵
    A = torch.cdist(X, X)
    B = torch.cdist(Y, Y)
    # 中心化距离矩阵
    A_centered = center_distance_matrix(A)
    B_centered = center_distance_matrix(B)
    # 计算距离协方差
    dCov = torch.sum(A_centered * B_centered) / (n * n)
    return dCov


def distance_variance(X):
    """
    计算距离方差
    :param X: 随机变量的样本
    :return: 距离方差
    """
    n = X.shape[0]
    # 计算距离矩阵
    A = torch.cdist(X, X)
    # 中心化距离矩阵
    A_centered = center_distance_matrix(A)
    # 计算距离方差
    dVar = torch.sum(A_centered ** 2) / (n * n)
    return dVar


def distance_correlation(X, Y):
    """
    计算距离相关性
    :param X: 第一个随机变量的样本
    :param Y: 第二个随机变量的样本
    :return: 距离相关性
    """
    dCov = distance_covariance(X, Y)
    dVarX = distance_variance(X)
    dVarY = distance_variance(Y)
    if dVarX == 0 or dVarY == 0:
        return 0
    dCor = torch.sqrt(dCov) / torch.sqrt(torch.sqrt(dVarX) * torch.sqrt(dVarY))
    return dCor.item()


# =====================================================================================================

def center_distance_matrix(D):
    """中心化距离矩阵"""
    A = D - torch.mean(D, dim=0, keepdim=True) - torch.mean(D, dim=1, keepdim=True) + torch.mean(D)
    return A


def distance_covariance(X, Y):
    """计算距离协方差"""
    n = X.shape[0]
    # 计算距离矩阵
    A = torch.cdist(X, X, p=2)
    B = torch.cdist(Y, Y, p=2)
    # 中心化距离矩阵
    A_centered = center_distance_matrix(A)
    B_centered = center_distance_matrix(B)
    # 计算距离协方差
    dCov = torch.sum(A_centered * B_centered) / (n * n)
    return dCov


def distance_correlation(X, Y):
    """计算距离相关性"""
    # 计算距离协方差
    dCov = distance_covariance(X, Y)
    # 计算距离方差
    dVarX = distance_covariance(X, X)
    dVarY = distance_covariance(Y, Y)
    # 计算距离相关性
    dCor = dCov / torch.sqrt(dVarX * dVarY)
    return dCor

# ======================================================================================================


def distance_correlation_torch(X, Y):
    n = X.shape[0]
    # 计算距离矩阵
    A = torch.cdist(X, X)
    B = torch.cdist(Y, Y)
    # 计算双中心距离矩阵
    A_mean_row = A.mean(dim=1, keepdim=True)
    A_mean_col = A.mean(dim=0, keepdim=True)
    A = A - A_mean_row - A_mean_col + A.mean()
    B_mean_row = B.mean(dim=1, keepdim=True)
    B_mean_col = B.mean(dim=0, keepdim=True)
    B = B - B_mean_row - B_mean_col + B.mean()
    # 计算距离协方差
    dCov = torch.sum(A * B) / (n * n)
    # 计算距离方差
    dVarX = torch.sum(A * A) / (n * n)
    dVarY = torch.sum(B * B) / (n * n)
    if dVarX == 0 or dVarY == 0:
        return 0.
    # 计算距离相关性
    # dCov = dCov / torch.sqrt(dVarX * dVarY)
    dCov = torch.sqrt(dCov) / torch.sqrt(torch.sqrt(dVarX) * torch.sqrt(dVarY))
    return dCov.item()


def distance_correlation2(x, y):
    n = x.size(0)
    # 计算距离矩阵
    a = torch.cdist(x, x)
    b = torch.cdist(y, y)
    # 中心化距离矩阵
    A_mean = a.mean(dim=0, keepdim=True)
    B_mean = b.mean(dim=0, keepdim=True)
    A_centered = a - A_mean - A_mean.t() + a.mean()
    B_centered = b - B_mean - B_mean.t() + b.mean()
    # 计算距离协方差
    dCovXY = torch.sqrt((A_centered * B_centered).sum() / (n * n))
    dVarX = torch.sqrt((A_centered * A_centered).sum() / (n * n))
    dVarY = torch.sqrt((B_centered * B_centered).sum() / (n * n))
    # 计算距离相关性
    dCor = dCovXY / torch.sqrt(dVarX * dVarY)
    return dCor.item()


def distance_correlation3(X, Y):
    n = X.shape[0]
    if X.dim() == 1:
        X = X.view(-1, 1)
    if Y.dim() == 1:
        Y = Y.view(-1, 1)
    # 计算距离矩阵
    A = torch.cdist(X, X)
    B = torch.cdist(Y, Y)
    # 中心化距离矩阵
    A_mean_rows = A.mean(dim=1, keepdim=True)
    A_mean_cols = A.mean(dim=0, keepdim=True)
    A_centered = A - A_mean_rows - A_mean_cols + A.mean()
    B_mean_rows = B.mean(dim=1, keepdim=True)
    B_mean_cols = B.mean(dim=0, keepdim=True)
    B_centered = B - B_mean_rows - B_mean_cols + B.mean()
    # 计算距离相关性
    dcov2_xy = (A_centered * B_centered).sum() / n**2
    dcov2_xx = (A_centered * A_centered).sum() / n**2
    dcov2_yy = (B_centered * B_centered).sum() / n**2
    dcor = torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))
    return dcor.item()


def distance_correlation4(X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    A = squareform(pdist(X, 'euclidean'))
    B = squareform(pdist(Y, 'euclidean'))
    A_mean = A.mean(axis=0, keepdims=True)
    B_mean = B.mean(axis=0, keepdims=True)
    A_centered = A - A_mean - A_mean.T + A.mean()
    B_centered = B - B_mean - B_mean.T + B.mean()
    dCovXY = np.sqrt((A_centered * B_centered).sum() / (n * n))
    dVarX = np.sqrt((A_centered * A_centered).sum() / (n * n))
    dVarY = np.sqrt((B_centered * B_centered).sum() / (n * n))
    return dCovXY / np.sqrt(dVarX * dVarY)


X = torch.randn((30, 100))
Y = torch.randn((30, 100))

# 计算距离相关性
print("Distance Correlation:")
# print(distance_correlation(X, Y))
print(distance_correlation_torch(X, Y))
print(distance_correlation2(X, Y))
print(distance_correlation3(X, Y))

print(dcor.distance_correlation(X.numpy(), Y.numpy()))
print(distance_correlation4(X.numpy(), Y.numpy()))