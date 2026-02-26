pub struct Matrix<T, const M: usize, const N: usize> where [(); M * N]: {
    pub data: [T; M * N]
}

impl<T: Copy + Clone + Default, const M: usize, const N: usize> Matrix<T, M, N> where [(); M * N]: {

    /// Produce a matrix with the default value of T
    pub fn zero() -> Self {
        Self {
            data: [T::default(); M * N]
        }
    }

    /// Construct from flat array based on generics
    pub fn build<A>(data: [A; M * N]) -> Matrix<T, M, N> where T: From<A> {
        Matrix {
            data: data.map(|x| x.into())
        }
    }

    /// Retrieve the index of a specific row and column
    #[inline]
    fn idx(row: usize, col: usize) -> usize {
        row * N + col
    }
}

impl<const M: usize> Matrix<f64, M, M> where [(); M * M]: {
    
    /// Yield the identity square matrix of MxM: f64
    pub fn identity() -> Self {

        let data = std::array::from_fn(|idx| {
            let col = idx % M;
            let row = idx / M;

            if col == row { 1f64 } else { 0f64 }
        });

        Self {
            data
        }
    }
}

impl<T: std::fmt::Debug, const M: usize, const N: usize> std::fmt::Debug
    for Matrix<T, M, N>
where
    [(); M * N]:,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Matrix<{}, {}> [", M, N)?;

        for r in 0..M {
            write!(f, "    [")?;
            for c in 0..N {
                let idx = r * N + c;
                write!(f, "{:?}", self.data[idx])?;

                if c < N - 1 {
                    write!(f, ", ")?;
                }
            }
            writeln!(f, "]")?;
        }

        write!(f, "]")
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

impl<T, const M: usize, const N: usize, const P: usize> std::ops::Mul<Matrix<T, N, P>> for &Matrix<T, M, N>
where [(); M * N]:, [(); M * P]:, [(); N * P]:, T: std::ops::Mul<Output = T> + std::iter::Sum + Copy
{
    type Output = Matrix<T, M, P>;

    fn mul(self, other: Matrix<T, N, P>) -> Matrix<T, M, P> {
        Matrix {
            data: std::array::from_fn(|idx| {
                let row = idx / P;
                let col = idx % P;
                (0..N).map(|val| self.data[row * N + val] * other.data[val * P + col])
                    .sum()
            })
        }
    }
}

impl<T, const M: usize, const N: usize, const P: usize> std::ops::Mul<Matrix<T, N, P>> for Matrix<T, M, N>
where [(); M * N]:, [(); M * P]:, [(); N * P]:, T: std::ops::Mul<Output = T> + std::iter::Sum + Copy
{
    type Output = Matrix<T, M, P>;

    fn mul(self, other: Matrix<T, N, P>) -> Matrix<T, M, P> {
        Matrix {
            data: std::array::from_fn(|idx| {
                let row = idx / P;
                let col = idx % P;
                (0..N).map(|val| self.data[row * N + val] * other.data[val * P + col])
                    .sum()
            })
        }
    }
}

impl<T, const M: usize, const N: usize> std::ops::Add<Matrix<T, M, N>> for Matrix<T, M, N>
where [(); M * N]:, T: std::ops::Add<Output = T> + Copy
{
    type Output = Matrix<T, M, N>;

    fn add(self, other: Matrix<T, M, N>) -> Matrix<T, M, N> {
        Matrix {
            data: std::array::from_fn(|idx| self.data[idx] + other.data[idx])
        }
    }
}

impl<T, const M: usize, const N: usize> std::ops::Add<Matrix<T, M, N>> for &Matrix<T, M, N>
where [(); M * N]:, T: std::ops::Add<Output = T> + Copy
{
    type Output = Matrix<T, M, N>;

    fn add(self, other: Matrix<T, M, N>) -> Matrix<T, M, N> {
        Matrix {
            data: std::array::from_fn(|idx| self.data[idx] + other.data[idx])
        }
    }
}

impl<T, const M: usize, const N: usize> std::ops::Sub<Matrix<T, M, N>> for Matrix<T, M, N>
where [(); M * N]:, T: std::ops::Sub<Output = T> + Copy
{
    type Output = Matrix<T, M, N>;

    fn sub(self, other: Matrix<T, M, N>) -> Matrix<T, M, N> {
        Matrix {
            data: std::array::from_fn(|idx| self.data[idx] - other.data[idx])
        }
    }
}

impl<T, V, const M: usize, const N: usize> std::ops::Mul<V> for Matrix<T, M, N>
where T: From<V> + std::ops::Mul<Output = T>, [(); M * N]:, V: Copy {
    type Output = Matrix<T, M, N>;

    fn mul(self, other: V) -> Matrix<T, M, N> {
        Matrix {
            data: self.data.map(|x| x * other.into())
        }
    }
}

impl<T, V, const M: usize, const N: usize> std::ops::Div<V> for Matrix<T, M, N>
where T: From<V> + std::ops::Div<Output = T>, [(); M * N]:, V: Copy {
    type Output = Matrix<T, M, N>;

    fn div(self, other: V) -> Matrix<T, M, N> {
        Matrix {
            data: self.data.map(|x| x / other.into())
        }
    }
}
