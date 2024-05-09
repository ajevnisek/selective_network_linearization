import math
import numpy as np
import torch

# device = torch.device(f"cuda") if torch.cuda.is_available() else 'cpu'

def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def centering_pytorch(K):
    device = K.device
    n = K.shape[0]
    unit = torch.ones([n, n], device=device)
    I = torch.eye(n, device=device)
    H = I - unit / n
    return (H @ K) @ H
    # return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def rbf_pytorch(X, sigma=None):
    # GX = np.dot(X, X.T)
    GX = X @ (X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = torch.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))

def kernel_HSIC_pytorch(X, Y, sigma):
    return torch.sum(centering_pytorch(rbf_pytorch(X, sigma)) * centering_pytorch(rbf_pytorch(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_HSIC_pytorch(X, Y):
    L_X = X @ (X.T)
    L_Y = Y @ (Y.T)
    return torch.sum(centering_pytorch(L_X) * centering_pytorch(L_Y))



def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def linear_CKA_pytorch(X, Y):
    hsic = linear_HSIC_pytorch(X, Y)
    var1 = torch.sqrt(linear_HSIC_pytorch(X, X))
    var2 = torch.sqrt(linear_HSIC_pytorch(Y, Y))

    return hsic / (var1 * var2)


def linear_CKA_pytorch_batched(X, Y):
    hsic = linear_HSIC_pytorch(X, Y)
    var1 = np.sqrt(linear_HSIC_pytorch(X, X))
    var2 = np.sqrt(linear_HSIC_pytorch(Y, Y))

    return hsic / (var1 * var2)



def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def kernel_CKA_pytorch(X, Y, sigma=None):
    hsic = kernel_HSIC_pytorch(X, Y, sigma)
    var1 = torch.sqrt(kernel_HSIC_pytorch(X, X, sigma))
    var2 = torch.sqrt(kernel_HSIC_pytorch(Y, Y, sigma))

    return hsic / (var1 * var2)


if __name__ == '__main__':
    X = np.random.randn(100, 64)
    Y = np.random.randn(100, 64)

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

    print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))

    hsic_x_x = []
    hsic_x_y = []
    hsic_x_z = []
    N_trails = 1000
    from math import sqrt
    nbins = int(sqrt(N_trails))
    for _ in range(N_trails):
        X = torch.randn((6, 2, 6))
        Y = torch.randn((6, 40))
        hsic_x_x.append(linear_CKA_pytorch(X.flatten(1), X.flatten(1)).item())
        hsic_x_y.append(linear_CKA_pytorch(X.flatten(1), Y).item())
        # gram_X = gram_linear_pytorch(X.flatten(1))
        # # gram_X = gram_linear_pytorch(X.flatten(1))
        # gram_Y = gram_linear_pytorch(Y)
        # gram_Y = gram_linear_pytorch(Y)
        # print(gram_X.shape,gram_Y.shape,)
        # hsic_x_x.append(hsic(gram_X, gram_X, unbiased=False, normalized=True).item())
        # hsic_x_y.append(hsic(gram_X, gram_Y, unbiased=False, normalized=True).item())

        W = torch.tensor([1, 1]).float()
        Z = (W @ X)
        # gram_Z = gram_linear_pytorch(Z)
        # gram_Z = gram_linear_pytorch(Z.flatten(1))
        # hsic_x_z.append(hsic(gram_X, gram_Z, unbiased=False, normalized=True).item())
        hsic_x_z.append(linear_CKA_pytorch(X.flatten(1), Z).item())
    import matplotlib.pyplot as plt
    plt.close('all')
    # plt.hist(hsic_x_x, bins=nbins, label='HSIC(X; Y)', alpha=0.8)
    plt.hist(hsic_x_y, bins=nbins, label='HSIC(X; Y)', alpha=0.8)
    plt.hist(hsic_x_z, bins=nbins, label='HSIC(X; Z)', alpha=0.8)
    plt.legend()
    plt.grid(True)
    x_shape = tuple(X.shape)
    y_shape = tuple(Y.shape)
    z_shape = tuple(Z.shape)
    plt.title(f'two histrograms of HSIC values\n'
              f'X,Y independent, and Z=W*X\n'
              f'[X]={x_shape}, [Y]={y_shape}, [Z]={z_shape}')
    plt.xlabel('HSIC')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig('hsic_distributions.png')
    print(hsic_x_x[0])

    X = np.random.randn(128, 32, 32, 64)
    Y = np.random.randn(128, 100)

    print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
    print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

    print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
    print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))