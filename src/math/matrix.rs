pub struct Matrix<T, const M: usize, const N: usize> where [(); M * N]: {
    data: [T; M * N]
}
