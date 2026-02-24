pub struct Matrix<T, const M: usize, const N: usize> where [(); M * N]: {
    data: [T; M * N]
}

impl<T: Copy + Clone + Default, const M: usize, const N: usize> Matrix<T, M, N> where [(); M * N]: {

    /// Produce a matrix with the default value of T
    pub fn zero() -> Self {
        Self {
            data: [T::default(); M * N]
        }
    }
}

impl<const M: usize> Matrix<f64, M, M> where [(); M * M]: {
    
    /// Yield the identity square matrix of MxM: f64
    pub fn identity() -> Self {

        let data = std::array::from_fn(|idx| {
            let col = (M * M) % idx;
            let row = (M * M) / idx;

            if col == row { 1f64 } else { 0f64 }
        });

        Self {
            data
        }
    }
}

impl<T: Copy + Clone + std::ops::Mul<Output = T>, const M: usize, const N: usize> Matrix<T, M, N> where [(); M * N]: {
    
    /// Elementwise multiplication / hadamard product / dot product
    pub fn dot(&self, other: &Matrix<T, M, N>) -> Matrix<T, M, N> {

        Self {
            data: std::array::from_fn(|idx| self.data[idx] * other.data[idx])
        }
    }
}
