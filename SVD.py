import glob
import numpy as np


def main():
    temp = [[1000, 100, 10], [10, 0, 80], [80, 50, 200]]
    mat = np.asarray(temp)
    s,v,d = np.linalg.svd(mat)
    s,v,d = np.round(s,decimals=4),np.round(v,decimals=4),np.round(d,decimals=4)
    a = np.round(s*v)
    print(np.round(a*d),"\n",mat)

if __name__ == '__main__':
    main()