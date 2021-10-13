use pyo3::types::PyFloat;
use pyo3::types::PyList;
use pyo3::{exceptions, prelude::*};
use rayon::prelude::*;
use std::collections::VecDeque;
mod rnumpy {

    pub struct DynamicList {
        pub els: Vec<f64>,
        pub shape: Vec<usize>,
    }

    impl<'source> FromPyObject<'source> for DynamicList {
        fn extract(ob: &'source PyAny) -> PyResult<Self> {
            let mut arr: Vec<f64> = Vec::new();
            let mut py_lists: VecDeque<&PyList> = VecDeque::new();
            py_lists.push_back(ob.downcast::<PyList>()?);

            let mut shape: Vec<usize> = Vec::new();
            let mut shape_counter = 0;

            while let Some(list) = py_lists.pop_front() {
                if shape_counter == 0 {
                    shape.push(list.len());
                    shape_counter = shape.iter().fold(1, |product, cur| cur * product) - 1;
                } else {
                    shape_counter -= 1;
                }

                for py_obj in list.iter() {
                    let name = py_obj.get_type().name()?;
                    match name {
                        "list" => py_lists.push_back(py_obj.downcast::<PyList>()?),
                        "float" => arr.push(py_obj.downcast::<PyFloat>()?.extract()?),
                        t => {
                            return Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                                "unsupported type: {}",
                                t
                            )))
                        }
                    }
                }
            }
            Ok(DynamicList { els: arr, shape })
        }
    }

    use super::*;
    #[pyclass]
    #[derive(FromPyObject)]
    pub struct Array {
        #[pyo3(get, set)]
        pub shape: Vec<usize>,
        #[pyo3(get, set)]
        pub stride: Vec<usize>,
        #[pyo3(get, set)]
        arr: Vec<f64>,
    }

    #[pymethods]
    impl Array {
        #[new]
        pub fn new(arr: DynamicList) -> Self {
            Array {
                shape: arr.shape.to_owned(),
                arr: arr.els.to_owned(),
                stride: vec![1],
            }
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

        pub fn add(&self, rhs: &Array) -> PyResult<Array> {
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
            Ok(Array {
                arr,
                stride: self.stride.clone(),
                shape: self.shape.clone(),
            })
        }
    }

    // fn arange(start: f64, end: f64, step: usize, shape: Vec<usize>) -> Array {
    //     let arr = (start..end).step_by(step).collect();
    //     Array::new(arr,shape )
    // }
}

#[pymodule]
fn rnumpy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<rnumpy::Array>()?;
    Ok(())
}
