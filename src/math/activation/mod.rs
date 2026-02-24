pub mod sigmoid;

pub trait Activation {
    fn calculate(value: f64) -> f64;
    fn derivative(value: f64) -> f64;
}
