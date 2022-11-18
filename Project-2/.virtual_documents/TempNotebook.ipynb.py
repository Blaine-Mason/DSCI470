import numpy as np
import numpy.linalg as la
from itertools import permutations, product
import math

def plusAndMinusPermutations(items):
    for p in permutations(items):
        for signs in product([-1, 1], repeat=len(items)):
            yield [a * sign for a, sign in zip(p, signs)]


def Frobenius_Norm(A, A_):
    return np.sqrt(np.trace(np.subtract(A, A_) @ np.subtract(A, A_).conj().T))


def radial_fb(X, val, q): # Set q to be the qth quantile
    return 1 / np.quantile((X.T).dot(val), q)


def compute_polar_fb(X, q, n):
    polar_body = np.ones((n,0))
    for i in range(X.shape[1]):
        if radial_fb(X, X[:,i] / la.norm(X[:,i]), q) >= la.norm(X[:,i]):
            yeet = np.array(X[:,i]).reshape(n,1)
            polar_body = np.hstack((polar_body, yeet))
    return polar_body


def compute_fb(X, q, n):
    Kpol = compute_polar_fb(X, q, n)
    floating_body = np.ones((n,0))
    maxVal = 0
    maxPoint = np.ones((n,0))
    for i in range(X.shape[1]):
        dotValMat = (X[:,i].T).dot(Kpol)
        for j in range(dotValMat.shape[0]):
            if dotValMat[j] > maxVal:
                maxVal = dotValMat[j]
        if maxVal <= 1:
            maxPoint = np.array(X[:,i]).reshape(n,1)
            floating_body = np.hstack((floating_body, maxPoint))
        maxVal = 0
    return floating_body


def Get_Min(A,A_):
    weights = -abs((A.T@A_)**2)
    ind = octave.munkres(weights)
    print(ind)
    fs = [int(x) - 1 for x in list(ind[0])]
    B = A_[:, fs]
    innerprod = np.diag(np.real(A.T@B) < 0)
    B[:,innerprod] = -B[:, innerprod]
    min_ = Frobenius_Norm(B, A)
    return min_


def ell_1(n):
    U = np.random.uniform(0,1,n)
    tmp = np.array([0,1])
    Final = np.zeros(n)
    U = np.hstack((U,tmp))
    U = np.sort(U)
    for i in range(1,n+1):
        Final[i-1] = U[i] - U[i-1]
    Final = np.reshape(Final, (n,1))
    for j in range(n):
        binom = np.random.binomial(1,.5)
        if(binom == 1):
            Final[j] = Final[j]*(-1)
    return Final


def simplex(n):
    U = np.random.uniform(0,1,n)
    tmp = np.array([0,1])
    Final = np.zeros(n)
    U = np.hstack((U,tmp))
    U = np.sort(U)
    for i in range(1,n+1):
        Final[i-1] = U[i] - U[i-1]
    Final = np.reshape(Final, (n,1))
    return Final

def moment(A, scale):
    temp = A
    samples = A.shape[1]
    thetas = np.arange(0, 2.1*math.pi, .1)
    moments = np.zeros(thetas.shape)
    lines = np.zeros((2, thetas.shape[0]))
    for i in range(thetas.shape[0]):
        vect = np.array([[np.sin(thetas[i]), np.cos(thetas[i])]])
        m = (1/samples) * np.sum((vect@temp)**1)
        moments[i] = m

    for i in range(thetas.shape[0]):
        x = moments[i]*np.cos(thetas[i])
        y = moments[i]*np.sin(thetas[i])
        lines[0,i] = scale*x
        lines[1,i] = scale*y
    return lines




