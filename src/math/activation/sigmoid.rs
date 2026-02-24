use crate::math::activation::Activation;

pub struct Sigmoid;

impl Activation for Sigmoid {

    // Standard 'sigmoid' logistic function
    fn calculate(value: f64) -> f64 {
        1f64 / (1f64 + (-value).exp())
    }

    // First order derivate of logistic function
    fn derivative(value: f64) -> f64 {
        let value = Self::calculate(value);
        value * (1f64 - value)
    }
}
