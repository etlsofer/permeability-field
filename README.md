# permeability-field
Found the conection between permeability field and conductivity field
# Data found so far for var = 1:
1. The distance between each transition matrix is ​​necessarily in the range of 3100-3300 according to norm 2.

2. The transition matrices are not reversible, however the subtraction matrices of the transition matrices and their determinant value is less than 10 in their absolute value.

3. The standard deviation of the values ​​in each transition matrix ranges from 70 to 81.

4. of the subtraction ranges from 102 to 116

5. In decomposing svd, very small values ​​are obtained in the s and v matrix (less than 0.2)

6. For the matrix d close values ​​are obtained between the matrices

7. The distance of the s matrices from each other according to norm 2 is very close and ranges from the values ​​24-25.

8. The distances of the v-matrix are relatively varied

9. Also the distances between the d matrices are very close and range between the same values ​​of 24-25

10. The distance between all the values ​​of the matrices s, d is constant and equal to 17.32 (a strange and interesting figure in itself)

As we will see then there is some connection between the transition matrices and the difference between them is mainly expressed in the v-matrix. It is therefore worth investigating the above matrix behavior.
In other words, there is a connection between the singular values ​​and the different results. This raises the question of whether it will be possible to find a matrix very close to all these matrices which preserve only the greatest self-values ​​and discard the smallest ones.
11. The largest singular value is relatively close between the matrices.
12. Even the smallest singular value is relatively close between the matrices.
Useful Links:
https://ai.stackexchange.com/questions/8058/machine-learning-to-predict-88-matrix-values-using-three-independent-matrices
Matrix prediction problem using a matrix.
http://proceedings.mlr.press/v23/hazan12b/hazan12b.pdf
an article on this subject
