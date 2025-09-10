# Logistic Regression's Training :
import numpy as np

# -------- Save weights and Cost ----------
def save_w_cost(W, Cost, fW='pesos.csv', fC='costo.csv'):
    np.savetxt(fW, W, delimiter=',')
    np.savetxt(fC, Cost.reshape(-1,1), delimiter=',')
    return

# -------- Initialize weights -------------
def iniWs(n_features):
    W = np.zeros((n_features, 1), dtype=float)
    V = np.zeros_like(W)
    return(W,V)

# -------- Load config --------------------
def conf_train(nFile):
    vals = np.loadtxt(nFile, delimiter=',', ndmin=1)

    global EPOCHS, ALPHA, PTRAIN
    EPOCHS = int(vals[0])
    ALPHA  = float(vals[1])
    PTRAIN = float(vals[2])

    # Si el porcentaje está en [0,1] → convertir a porcentaje real
    if 0 < PTRAIN <= 1:
        PTRAIN = int(round(PTRAIN * 100))
    else:
        PTRAIN = int(PTRAIN)

    # Validación mínima (como pide el profesor: 60<p<81)
    if not (60 <= PTRAIN <= 81):
        print(f"[ADVERTENCIA] Porcentaje {PTRAIN} fuera de rango (60-81).")

    return

# -------- Load data ----------------------
def load_data():
    X = np.loadtxt('dfeatures.csv', delimiter=',')
    y = np.loadtxt('label.csv', delimiter=',').reshape(-1,1)

    # barajar datos
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # añadir bias
    Xb = np.hstack([X, np.ones((X.shape[0],1))])

    # split train/test
    N = Xb.shape[0]
    L = int(round(N * (PTRAIN/100.0)))
    Xtrn, Ytrn = Xb[:L], y[:L]
    Xtst, Ytst = Xb[L:], y[L:]

    # guardar archivos
    np.savetxt('dtrn.csv',       Xtrn, delimiter=',')
    np.savetxt('dtrn_label.csv', Ytrn, delimiter=',')
    np.savetxt('dtst.csv',       Xtst, delimiter=',')
    np.savetxt('dtst_label.csv', Ytst, delimiter=',')

    # exponer como globales
    global X_trn, y_trn, X_tst, y_tst
    X_trn, y_trn, X_tst, y_tst = Xtrn, Ytrn, Xtst, Ytst
    return

# -------- Training -----------------------
def train():
    N, d = X_trn.shape
    W, V = iniWs(d)
    Cost = np.zeros((EPOCHS,1), dtype=float)

    for k in range(EPOCHS):
        z  = X_trn @ W
        y  = 1.0/(1.0 + np.exp(-z))   # sigmoide

        # costo
        J  = -(1.0/N)*np.sum(y_trn*np.log(y+1e-12) + (1-y_trn)*np.log(1-y+1e-12))
        g  = (1.0/N)*(X_trn.T @ (y-y_trn))   # gradiente

        V  = 0.9*V + ALPHA*g   # momentum fijo en 0.9
        W  = W - V

        Cost[k,0] = J

        # debug cada 1000 iteraciones
        if k % 1000 == 0:
            preds = (y >= 0.5).astype(int)
            acc   = np.mean(preds == y_trn)
            print(f"[Iter {k}] Costo={J:.4f}, Acc={acc:.2f}")

    global W_opt, Cost_vec
    W_opt, Cost_vec = W, Cost
    return

# -------- Main ---------------------------
def main():
    conf_train('config/conf_train.csv')
    load_data()
    train()
    save_w_cost(W_opt, Cost_vec, 'pesos.csv','costo.csv')

if __name__ == '__main__':
    main()
