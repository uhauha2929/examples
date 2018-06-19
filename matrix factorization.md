## LU分解
---

```{.python .input  n=15}
def LU(A):
    U = np.copy(A)
    m, n = A.shape
    # 注意L一开始是Identity
    L = np.eye(n)

    # 最后一列(n)不用处理
    for k in range(n-1):
        # 第1行不用处理
        for j in range(k+1,n):
            L[j,k] = U[j,k]/U[k,k]
            U[j,k:n] -= L[j,k] * U[k,k:n]
            #print(U)会发现U就是在做高斯消元
            print(U)
    return L, U
```

```{.python .input  n=16}
import numpy as np
A = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]], dtype=np.float64)
L, U = LU(A)
print(L)
print(U)
```

```{.json .output n=16}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[ 1.  2.  3.]\n [ 0. -3. -6.]\n [ 7.  8.  9.]]\n[[  1.   2.   3.]\n [  0.  -3.  -6.]\n [  0.  -6. -12.]]\n[[ 1.  2.  3.]\n [ 0. -3. -6.]\n [ 0.  0.  0.]]\n[[1. 0. 0.]\n [4. 1. 0.]\n [7. 2. 1.]]\n[[ 1.  2.  3.]\n [ 0. -3. -6.]\n [ 0.  0.  0.]]\n"
 }
]
```
