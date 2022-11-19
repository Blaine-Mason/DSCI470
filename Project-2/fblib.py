import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from munkres import Munkres
from tqdm import tqdm

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

def obtain_samples_lp(samples, n, p_):
    S = np.zeros((n, 1))
    for i in range(samples):
        choice = np.random.choice(2, 1, p=[p_, 1-p_])
        if(choice == 0):
            S = np.hstack((S, ell_1(n)))
        if(choice == 1):
            S = np.hstack((S,np.random.standard_cauchy((n,1))))
            #S = np.hstack((S, np.random.pareto(3,(n,1))))
    S = np.delete(S, 0,1)
    return S

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


def obtain_samples_simplex(samples, n, p_):
    S = np.zeros((n, 1))
    for i in range(samples):
        choice = np.random.choice(2, 1, p=[p_, 1-p_])
        if(choice == 0):
            S = np.hstack((S, simplex(n)))
        if(choice == 1):
            S = np.hstack((S,np.random.standard_cauchy((n,1))))
            #S = np.hstack((S, np.random.pareto(3,(n,1))))
    S = np.delete(S, 0,1)
    mean = np.mean(S)
    S = S - mean
    return S, mean


def Frobenius_Norm(A, A_):
    return np.sqrt(np.trace(np.subtract(A, A_)@np.subtract(A,A_).conj().T))


def Get_Min(A,A_):
    weights = -abs((A.T@A_)**2)
    m = Munkres()
    ind = m.compute(weights)
    ind = [[x[1] for x in ind]]
    B = A_[:, ind[0]]
    innerprod = np.diag(np.real(A.T@B) < 0)
    B[:,innerprod] = -B[:, innerprod]
    min_ = Frobenius_Norm(B, A)
    return min_ 


def llp(n, percent, samples, fb):
    S = obtain_samples_lp(samples, n, percent)
    A = np.random.normal(0,1,(n,n))
    sqrt = np.sqrt(np.sum(A**2, axis = 0))
    A = A  / (np.tile(sqrt,(n,1)))
    #A = np.array([[1,0],[0,1]])
    X = A@S
    if fb:
        X = compute_fb(X,.95,n)
        num_of_samples_left = samples - X.shape[1]
    T = np.random.gamma(((X.shape[0]) / 1) + 1, 1, (1, X.shape[1]))
    Q = X * np.tile(T ** (1 / 1), (X.shape[0], 1))
    model = FastICA(n_components=3)
    transform = model.fit_transform(Q.T)
    M_til = model.mixing_
    A_til = np.copy(M_til)
    
    if fb:
        return Get_Min(A, A_til), num_of_samples_left
    else:
        return Get_Min(A, A_til)


def lls(n, percent, samples, plots, fb):
    S, mean = obtain_samples_simplex(samples, n, percent)
    V = np.hstack((np.zeros((n,1)),np.eye(n,n))) - mean
    A = np.random.normal(0,1,(n,n))
    sqrt = np.sqrt(np.sum(A**2, axis = 0))
    A = A  / (np.tile(sqrt,(n,1)))
    
    #A = np.array([[1,0],[0,1]])
    X = A@S
    V = A@V
    if fb:
        X = compute_fb(X,.95,n)
        num_of_samples_left =  samples - X.shape[1]
        
    X = np.vstack([X, np.ones((1,X.shape[1]))])
    
    T = np.random.gamma(X.shape[0], 1, (1,X.shape[1]))
    Q = X * np.tile(T,(X.shape[0],1))
    model = FastICA(n_components=n+1)
    transform = model.fit_transform(Q.T)
    A_EST = model.mixing_
    sign = 1 / A_EST[-1,:]
    arr = (A_EST) * np.tile(sign,(A_EST.shape[0], 1))
    verts = arr[0:-1,:]
    verts = verts/la.norm(verts,2)
    if plots:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.subplot(2,2,1)
        plt.plot(S[0,:], S[1,:], "or", markersize = 0.3)
        plt.subplot(2,1,2)
        plt.plot(X[0,:], X[1,:], "or", markersize = 0.3)
        plt.plot(verts[0,:], verts[1,:], "gs", markersize = 5)
        plt.subplot(2,2,2)
        plt.plot(Q[0,:], Q[1, :], "og", markersize = .3)
        plt.show()
    if fb:
        return Get_Min(V/la.norm(V,2), verts/la.norm(verts,2)), num_of_samples_left
    else:
        return Get_Min(V/la.norm(V,2), verts/la.norm(verts,2))


def support(X, val, q): # Set q to be the qth quantile
    return np.quantile((X.T).dot(val), q)

def compute_fb(X, q, n):
    polar_body = np.ones((n,0))
    for i in range(X.shape[1]):
        if support(X, X[:,i] / la.norm(X[:,i]), q) > np.dot(X[:,i], X[:,i]/la.norm(X[:,i])): 
            polar_body = np.hstack((polar_body, np.array(X[:,i]).reshape(n,1)))
    return polar_body

def get_data(n, percent, a, b, step, plts, fb, shape, avg):
    ret_error = np.zeros(len(range(a,b,step)))
    if fb: ret_samples_removed = np.zeros(len(range(a,b,step)))
    index = 0
    for samples in tqdm(range(a, b, step)):
        sum_of_error = 0
        if fb: sum_of_samples = 0
        for i in range(avg):
            if fb:
                samples_removed = 0
                while samples_removed == 0:
                    if shape == "simplex":
                        err, samples_removed = lls(n, percent, samples, plts, fb)
                    elif shape == "lp":
                        err, samples_removed = llp(n, percent, samples, fb)
                    else:
                        print("Incorrect shape given")
                        return 0;
                sum_of_samples += samples_removed
                sum_of_error += err
            else:
                if shape == "simplex":
                    err = lls(n, percent, samples, plts, fb)
                elif shape == "lp":
                    err = llp(n, percent, samples, fb)
                else:
                    print("Incorrect shape given")
                    return 0
                sum_of_error += err
        if fb:
            ret_error[index] = sum_of_error/avg
            ret_samples_removed[index] = sum_of_samples/avg
            #print(f"Error: {sum_of_error/avg}, Samples Removed: {sum_of_samples/avg}")
        else:
            ret_error[index] = sum_of_error/avg
            #print(f"Error: {sum_of_error/avg}")
        index += 1
        #print(f"{samples} samples done.")
    return ret_error