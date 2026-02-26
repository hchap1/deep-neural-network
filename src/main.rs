#![allow(clippy::identity_op)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod math;
mod networks;

use networks::multilayer_sigmoidal_perceptron::Mlp;
use crate::networks::multilayer_sigmoidal_perceptron::Layer;

fn main() {
    let mlp = Mlp::<4>::new(
        vec![
            Box::new(Layer::<20, 40>::xavier_sigmoid()),
            Box::new(Layer::<40, 30>::xavier_sigmoid()),
            Box::new(Layer::<30, 20>::xavier_sigmoid()),
        ]
    );
}
