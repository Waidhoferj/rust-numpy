use pyo3::{exceptions, prelude::*};
use rayon::prelude::*;
#[pyclass]
#[derive(FromPyObject)]
pub struct Rnp {
    #[pyo3(get, set)]
    pub shape: Vec<usize>,
    #[pyo3(get, set)]
    pub stride: Vec<usize>,
    #[pyo3(get, set)]
    arr: Vec<f64>,
}

#[pymethods]
impl Rnp {
    #[new]
    pub fn new(arr: Vec<f64>, shape: Vec<usize>) -> Self {
        let stride: Vec<usize> = (0..shape.len())
            .map(|i| shape[i..].iter().fold(0, |acc, v| v * acc))
            .collect();
        Rnp { shape, arr, stride }
    }

    pub fn resize(&mut self, shape: Vec<usize>) -> PyResult<()> {
        let len = self.arr.len();
        let can_reshape = shape.iter().fold(Some(len), |dim, divisor| match dim {
            Some(val) if val % divisor == 0 => Some(val / divisor),
            _ => None,
        });
        if let Some(_) = can_reshape {
            self.shape = shape;
            Ok(())
        } else {
            Err(PyErr::new::<exceptions::PyTypeError, _>("Error message"))
        }
    }

    pub fn add(&self, rhs: &Rnp) -> PyResult<Rnp> {
        let same_shape = self
            .shape
            .par_iter()
            .zip(rhs.shape.par_iter())
            .all(|(s1, s2)| s1 == s2);

        if !same_shape {
            return Err(PyErr::new::<exceptions::PyTypeError, _>("Not same shape"));
        }
        let arr = self
            .arr
            .par_iter()
            .zip(self.arr.par_iter())
            .map(|(l, r)| l + r)
            .collect();
        Ok(Rnp {
            arr,
            stride: self.stride.clone(),
            shape: self.shape.clone(),
        })
    }
}

#[pymodule]
fn rnumpy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Rnp>()?;
    Ok(())
}
