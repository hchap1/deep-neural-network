use crate::math::matrix::Matrix;
use rand::distr::Distribution;
use rand::{distr::Uniform, rng};

pub trait Forward {
    fn feedforward(&self, input: &[f64], output: &mut [f64]);
    fn output_size(&self) -> usize;
    fn input_size(&self) -> usize;
}

pub struct Layer<const L: usize, const C: usize>
where
    [(); C * 1]:,
    [(); C * L]:,
{
    weights: Matrix<f64, C, L>,
    biases: Matrix<f64, C, 1>,
}

impl<const L: usize, const C: usize> Layer<L, C>
where
    [(); C * 1]:,
    [(); C * L]:,
    [(); L * 1]:,
{
    pub fn xavier_sigmoid() -> Self {
        let mut rng = rng();
        let fan_in = C as f64;
        let fan_out = L as f64;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();
        let dist = Uniform::new(-limit, limit).expect("Unable to create uniform distribution");
        let data = std::array::from_fn(|_| dist.sample(&mut rng));
        Self {
            weights: Matrix::build(data),
            biases: Matrix::zero(),
        }
    }

    pub fn feedforward(&self, previous: Matrix<f64, L, 1>) -> Matrix<f64, C, 1> {
        &self.biases + &self.weights * previous
    }
}

impl<const L: usize, const C: usize> Forward for Layer<L, C>
where
    [(); C * 1]:,
    [(); C * L]:,
    [(); L * 1]:,
{
    fn feedforward(&self, input: &[f64], output: &mut [f64]) {
        assert_eq!(input.len(), L, "Input size mismatch: expected {L}, got {}", input.len());
        assert_eq!(output.len(), C, "Output size mismatch: expected {C}, got {}", output.len());
        let matrix = Matrix::<f64, L, 1>::build(std::array::from_fn(|i| input[i]));
        let result = self.feedforward(matrix);
        output.copy_from_slice(&result.data);
    }

    fn output_size(&self) -> usize { C }
    fn input_size(&self) -> usize { L }
}

pub struct Mlp<const BUF: usize> {
    layers: Vec<Box<dyn Forward>>,
    buf_a: [f64; BUF],
    buf_b: [f64; BUF],
}

impl<const BUF: usize> Mlp<BUF> {
    pub fn new(layers: Vec<Box<dyn Forward>>) -> Self {
        debug_assert!(
            layers.iter().map(|l| l.output_size()).max().unwrap_or(0) <= BUF,
            "BUF is too small for the largest layer output"
        );
        Self {
            layers,
            buf_a: [0.0; BUF],
            buf_b: [0.0; BUF],
        }
    }

    pub fn feedforward(&mut self, input: &[f64], output: &mut [f64]) {
        self.buf_a[..input.len()].copy_from_slice(input);

        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            let in_size = layer.input_size();
            let out_size = layer.output_size();
            if i == last {
                layer.feedforward(&self.buf_a[..in_size], output);
            } else {
                let (src, dst) = (&self.buf_a[..in_size], &mut self.buf_b[..out_size]);
                let src = unsafe { &*(src as *const [f64]) };
                layer.feedforward(src, dst);
                self.buf_a[..out_size].copy_from_slice(&self.buf_b[..out_size]);
            }
        }
    }
}
