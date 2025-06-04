import numpy as np

def meancenter(images):
    mean_face = np.mean(images, axis=0)
    centered = images - mean_face
    return centered, mean_face

def h_matrikscov(data):
    return np.dot(data, data.T)

def pow_iter(A, num_iter=1000, tol=1e-6):
    b_k = np.random.rand(A.shape[1])
    b_k /= np.linalg.norm(b_k)

    for _ in range(num_iter):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        if b_k1_norm == 0:
            break
        b_k1 /= b_k1_norm
        if np.linalg.norm(b_k - b_k1) < tol:
            break
        b_k = b_k1

    eigenvalue = np.dot(b_k.T, np.dot(A, b_k))
    return eigenvalue, b_k

def h_eigenvekt(data, num_components=None):
    cov_matrix = h_matrikscov(data)

    eigenvalues = []
    eigenvectors = []

    B = cov_matrix.copy()

    n_components = num_components if num_components else data.shape[0]

    for _ in range(n_components):
        val, vec = pow_iter(B)
        eigenvalues.append(val)
        eigenvectors.append(vec)

        B = B - val * np.outer(vec, vec)

    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors).T  

    eigenfaces = np.dot(data.T, eigenvectors)

    for i in range(eigenfaces.shape[1]):
        norm = np.linalg.norm(eigenfaces[:, i])
        if norm > 0:
            eigenfaces[:, i] /= norm

    return eigenfaces, eigenvalues
