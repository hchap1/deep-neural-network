#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod math;

use math::matrix::Matrix;

fn main() {
    let a = Matrix::<f64, 2, 3>::build([
        1, 2, 3,
        4, 5, 6
    ]);

    let b = Matrix::<f64, 3, 2>::build([
        7, 8,
        9, 10,
        11, 12
    ]);

    println!("AxB =\n{:?}", a * b);

    println!("IDENTITY:\n{:?}", Matrix::<f64, 5, 5>::identity());
}
