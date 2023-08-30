import basis
import numpy as np
import matplotlib.pyplot as plt

dft_basis = basis.dft_matrix(32)

index_1 = 0
index_2 = 3
dot_prod = np.abs(np.dot(dft_basis[index_1,:], dft_basis[index_2,:]))
print("The dot product of the indexes {:d} and {:d}: {:4f}".format(index_1+1, index_2+1, dot_prod))

index = 3
l2_len = np.abs(np.linalg.norm(dft_basis[index,:],2))
print("The Euclidean lenght of the index {:d}: {:4f}".format(index+1, l2_len))

kronecker_basis = basis.kronecker_matrix(32)

mutcoh = basis.coherence(dft_basis, kronecker_basis)
print("The mutual coherence value between the measurement and the transform domains: {:4f}".format(mutcoh))