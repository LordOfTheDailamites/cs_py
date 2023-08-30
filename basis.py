"""
basis.py aims to include the transform matrices of the well-known bases used in 
sparse encoding, as well as available common measurement matrices and measure units

The aim is not to introduce faster computational algorithms, but rather implement
the available transform matrices

Author: Cemre Ömer Ayna
"""

import numpy as np

"""
    n x n orthonormal representation bases
    The matrix size n should be a power of 2.
"""
# dft_matrix creates an n x n discrete Fourier transform matrix.
def dft_matrix(size, normalize=True):
    if np.log2(size) % 1 > 0:   # Raise error if size is not a power of 2.
        raise Exception("Error: n should be powers of 2")
    col, row = np.meshgrid(np.arange(size), np.arange(size))
    dft = np.exp(-2 * np.pi * 1j/size * col * row)
    if normalize == True:   # Return a normalized transform matrix if normalize is True.
        return dft / np.sqrt(size)
    return dft

# dct_matrix creates an n x n discrete cosine transform matrix.
def dct_matrix(size, normalize=True):
    if np.log2(size) % 1 > 0:   # Raise error if size is not a power of 2.
        raise Exception("Error: n should be powers of 2")
    col, row = np.meshgrid(np.arange(size), np.arange(size))
    dct = np.cos(np.pi/size * (col + 0.5) * row)
    if normalize == True:   # Return a normalized transform matrix if normalize is True.
        dct[1:,:] = dct[1:,:] * np.sqrt(2)
        return dct / np.sqrt(size)
    return dct

# haar_matrix creates an n x n Haar transform matrix.
def haar_matrix(size, normalize=True):
    if np.log2(size) % 1 > 0:   # Raise error if size is not a power of 2.
        raise Exception("Error: n should be powers of 2")
    haar = np.array([[1]])
    scale = int(np.log2(size))
    for i in range(scale):
        if normalize == True:   # Scale the transform matrix with normalization.
            haar = np.concatenate((np.kron(haar, [1,1]), np.kron(np.eye(2**i), [1,-1])), axis=0) /np.sqrt(2)
        else:                   # Scale the transform matrix without normalization.
            haar = np.concatenate((np.kron(haar, [1,1]), np.kron(np.eye(2**i), [1,-1])), axis=0)
    return haar

"""
    n x n orthonormal measurement bases
    The matrix size n should be a power of 2.
"""

# kronecker_matrix returns an n x n Kroenecker transform matrix. It is naturally orthonormal.
def kronecker_matrix(n):
    return np.eye(n)

# kronecker_matrix returns an n x n noiselet transform matrix.
def noiselet_matrix(n):
    if np.log2(n) % 1 > 0:
        raise Exception("Error: n should be powers of 2")
    noiselet = np.concatenate((np.ones(shape=(n, 1)), np.zeros(shape=(n, 2*n-2))), axis=1)
    vec = np.zeros(shape=(int(n/2), 1))
    for col in range(n-1):
        vec_2x = np.concatenate((noiselet[1:n:2,col,np.newaxis],vec), axis=0)
        vec_2x_1 = np.concatenate((vec, noiselet[1:n:2,col,np.newaxis]), axis=0)
        noiselet[:,2*col] = (1 - 1j)*vec_2x + (1 + 1j)*vec_2x_1 
        noiselet[:,2*col+1] = (1 + 1j)*vec_2x + (1 - 1j)*vec_2x_1         
    return noiselet[:,n-1:]/n

"""
    common units
"""
# Mutual Coherence value
def coherence(mes, rep):
    mu = 0
    n = mes.shape[0]
    for i in range(n):
        for j in range(n):
            mu = np.maximum(mu, np.sqrt(n) * np.dot(mes[i,:],rep[:,j]))
    return np.abs(mu)

# PSNR value between two signals
def psnr(im1, im2):
    return 20*np.log10(255/np.sqrt(np.square(im1 - im2).mean()))
