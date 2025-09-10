# Testing for Logistic Regression
import numpy as np

def forward(xv,W):
    z=xv@W
    y=1.0/(1.0+np.exp(-z))
    return (y>=0.5).astype(int)

def measure(yv,zv):
    y=yv.astype(int).ravel(); z=zv.astype(int).ravel()
    cm=np.zeros((2,2),dtype=int)
    for yi,zi in zip(y,z):
        cm[yi,zi]+=1
    Fs=np.zeros((1,2))
    for c in (0,1):
        tp=cm[c,c]; fp=cm[1-c,c]; fn=cm[c,1-c]
        prec=tp/(tp+fp+1e-12); rec=tp/(tp+fn+1e-12)
        Fs[0,c]=2*prec*rec/(prec+rec+1e-12)
    return cm,Fs

metricas=measure

def save_measure(cm,Fsc,f1,f2):
    np.savetxt(f1,cm,fmt='%d',delimiter=',')
    np.savetxt(f2,Fsc,fmt='%.6f',delimiter=',')

def load_w(nFile='pesos.csv'):
    return np.loadtxt(nFile,delimiter=',').reshape(-1,1)

def load_data():
    X=np.loadtxt('dtst.csv',delimiter=',')
    y=np.loadtxt('dtst_label.csv',delimiter=',').reshape(-1,1)
    return X,y

# Beginning ...
def main():
    xv,yv=load_data()
    W=load_w()
    zv=forward(xv,W)
    cm,Fsc=metricas(yv,zv)
    save_measure(cm,Fsc,'cmatrix.csv','fscores.csv')

if __name__ == '__main__':   
    main()
