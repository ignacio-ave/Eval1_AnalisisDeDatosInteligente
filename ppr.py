#----------------------------------------------
# Create Features by use 
# Dispersion Entropy and Permutation entropy
#----------------------------------------------

import pandas as pd
import numpy  as np
from utility import entropy_dispersion, entropy_permuta

_X   = None
W    = None
_F1  = None
_F2  = None

# Load parameters Entropy
def conf_entropy():    
    conf = pd.read_csv("config/conf_ppr.csv", header=None)
    opt_code = str(conf.iloc[0,0]).strip()
    if opt_code == "1":
        opt = "dispersion"
    elif opt_code == "2":
        opt = "permutation"
    else:
        raise ValueError("Código de entropía inválido (1=dispersion, 2=permutation)")
    d   = int(conf.iloc[1,0])
    tau = int(conf.iloc[2,0])
    c   = int(conf.iloc[3,0])
    global W
    W = int(conf.iloc[4,0]) if conf.shape[0] >= 5 else None
    return(opt,d,tau,c)

# Load Data
def load_data(nFile):
    global _X
    _X = pd.read_csv(nFile, header=None).values
    return _X

# Obtain entropy : dispersión and Permutation
def gets_entropy(x,opt,d,tau,c):
    if opt == "dispersion":
        return entropy_dispersion(x,d,tau,c)
    elif opt == "permutation":
        return entropy_permuta(x,d,tau)
    else:
        raise ValueError("Método inválido")

# Obtain Features by use Entropy    
def gets_features():
    if _X is None:
        raise RuntimeError("No hay datos cargados. Llame primero a load_data().")
    X = _X
    N,L = X.shape
    W_eff = W if (W is not None and 1 <= W <= N) else N
    K = N // W_eff
    opt,d,tau,c = conf_entropy()
    F_rows = []
    base = 0
    for _ in range(K):
        seg = X[base:base+W_eff,:]
        feats = [ gets_entropy(seg[:,j],opt,d,tau,c) for j in range(L) ]
        F_rows.append(feats)
        base += W_eff
    return np.array(F_rows,dtype=float)

def save_data(F):
    global _F1,_F2
    pd.DataFrame(_F1).to_csv("dfeatures1.csv",header=False,index=False)
    pd.DataFrame(_F2).to_csv("dfeatures2.csv",header=False,index=False)
    pd.DataFrame(F).to_csv("dfeatures.csv",header=False,index=False)
    K1,K2=_F1.shape[0],_F2.shape[0]
    Y=np.concatenate([np.ones((K1,1)),np.zeros((K2,1))],axis=0)
    pd.DataFrame(Y).to_csv("label.csv",header=False,index=False)

# Beginning ...
def main():
    conf_entropy()
    load_data("data/class1.csv")
    F1 = gets_features()
    load_data("data/class2.csv")
    F2 = gets_features()
    F  = np.concatenate((F1,F2),axis=0)
    global _F1,_F2
    _F1,_F2 = F1,F2
    save_data(F)

if __name__ == '__main__':
    main()
