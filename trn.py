# Logistic Regression's Training :

import numpy      as np
import utility    as ut

#Save weights and Cost
def save_w_cost(W, Cost, fW='pesos.csv', fC='costo.csv'):
    np.savetxt(fW, W, delimiter=',')
    np.savetxt(fC, Cost.reshape(-1,1), delimiter=',')
    return

#
def iniWs(n_features):
    W = np.zeros((n_features, 1), dtype=float)
    V = np.zeros_like(W)
    return(W,V)

# Load data to train 
def load_data():
    # Cargar dfeatures.csv y label.csv; barajar y dividir según p (global)
    X = np.loadtxt('dfeatures.csv', delimiter=',')
    y = np.loadtxt('label.csv',    delimiter=',').reshape(-1,1)

    # Barajar de forma reproducible (si se desea, fijar semilla afuera)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Añadir bias a la derecha
    Xb = np.hstack([X, np.ones((X.shape[0],1))])

    # Partición entrenamiento / prueba
    N = Xb.shape[0]
    L = int(round(N * (ptrain/100.0)))
    Xtrn, Ytrn = Xb[:L], y[:L]
    Xtst, Ytst = Xb[L:], y[L:]

    # Guardar archivos solicitados por la etapa
    np.savetxt('dtrn.csv',       Xtrn, delimiter=',')
    np.savetxt('dtrn_label.csv', Ytrn, delimiter=',')
    np.savetxt('dtst.csv',       Xtst, delimiter=',')
    np.savetxt('dtst_label.csv', Ytst, delimiter=',')

    # Exponer como globales (perfil purista del profesor)
    global X_trn, y_trn, X_tst, y_tst
    X_trn, y_trn, X_tst, y_tst = Xtrn, Ytrn, Xtst, Ytst
    return()

#
#Training by use mGD
def train():    
    # Usa X_trn, y_trn, alpha, beta, epochs (globales)
    N, d = X_trn.shape
    W, V = iniWs(d)

    Cost = np.zeros((epochs,1), dtype=float)

    for k in range(epochs):
        z  = X_trn @ W
        y  = 1.0/(1.0 + np.exp(-z))              # sigmoide
        J  = - (1.0/N) * np.sum( y_trn*np.log(y+1e-12) + (1-y_trn)*np.log(1-y+1e-12) )
        g  = (1.0/N) * (X_trn.T @ (y - y_trn))   # gradiente
        V  = beta*V + alpha*g
        W  = W - V
        Cost[k,0] = J

    # Exponer como globales para respetar main() dado
    global W_opt, Cost_vec
    W_opt, Cost_vec = W, Cost
    return()

#
def conf_train(nFile):
    # Formato vertical esperado: alpha, beta, epochs, ptrain
    vals = np.loadtxt(nFile, delimiter=',', ndmin=1)
    # Aceptar 1D o columna
    if vals.ndim == 1:
        alpha_v  = float(vals[0]) if len(vals) > 0 else 0.1
        beta_v   = float(vals[1]) if len(vals) > 1 else 0.9
        epochs_v = int(  vals[2]) if len(vals) > 2 else 200
        ptrain_v = int(  vals[3]) if len(vals) > 3 else 80
    else:
        alpha_v  = float(vals[0,0]) if vals.shape[0] > 0 else 0.1
        beta_v   = float(vals[1,0]) if vals.shape[0] > 1 else 0.9
        epochs_v = int(  vals[2,0]) if vals.shape[0] > 2 else 200
        ptrain_v = int(  vals[3,0]) if vals.shape[0] > 3 else 80

    # Limpiar y exponer como globales (perfil del profesor)
    global alpha, beta, epochs, ptrain
    alpha  = max(1e-6, alpha_v)
    beta   = min(0.9999, max(0.0, beta_v))
    epochs = max(1, int(epochs_v))
    ptrain = min(100, max(1, int(ptrain_v)))
    return()

# Beginning ...
def main():    
    conf_train('config/conf_train.csv')
    load_data()   
    train()             
    save_w_cost(W_opt, Cost_vec, 'pesos.csv','costo.csv')
       
if __name__ == '__main__':   
	 main()



"""
# Logistic Regression's Training :

import numpy      as np
import utility    as ut

#Save weights and Cost
def save_w_cost():
    ...
    return
#
def iniWs():
    ...
    return(W,V)
# Load config train for Regression
#
#Training by use mGD
def train():    
    
    ....
    return()
# Load data to train 
def load_data():
    
    
    return()
#
def conf_train(nFile):
    ....
    return()

# Beginning ...
def main():    
    conf_train()
    load_data()   
    train()             
    save_w_cost(W,Cost, 'pesos.csv','costo.csv')
       
if __name__ == '__main__':   
	 main()

"""