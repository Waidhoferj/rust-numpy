use pyo3::types::{PyFloat, PyList};
use pyo3::{exceptions, prelude::*, PyNumberProtocol};
use std::collections::VecDeque;
pub mod rnumpy {

    use super::*;

    #[pyclass]
    pub struct Array {
        #[pyo3(get, set)]
        pub shape: Vec<usize>,
        #[pyo3(get, set)]
        arr: Vec<f64>,
    }

    #[pymethods]
    impl Array {
        #[new]
        pub fn new(arr: Self) -> Self {
            arr
        }

        pub fn reshape(&self, shape: Vec<usize>) -> PyResult<Array> {
            let len = self.arr.len();
            let can_reshape = shape.iter().fold(Some(len), |dim, divisor| match dim {
                Some(val) if val % divisor == 0 => Some(val / divisor),
                _ => None,
            });
            if let Some(_) = can_reshape {
                Ok(Array {
                    arr: self.arr.clone(),
                    shape,
                })
            } else {
                Err(PyErr::new::<exceptions::PyTypeError, _>("Error message"))
            }
        }
    }

    impl<'source> FromPyObject<'source> for Array {
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
            Ok(Self { arr, shape })
        }
    }

    #[pyproto]
    impl PyNumberProtocol for Array {
        fn __add__(lhs: Array, rhs: Array) -> PyResult<Array> {
            if lhs.shape != rhs.shape {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                    "arrays don't have the same shape: {:#?} vs {:#?}",
                    lhs.shape, rhs.shape
                )));
            }
            let arr: Vec<f64> = lhs
                .arr
                .iter()
                .zip(rhs.arr.iter())
                .map(|(l, r)| l + r)
                .collect();
            Ok(Array {
                arr,
                shape: lhs.shape.clone(),
            })
        }

        fn __neg__(self) -> PyResult<Array> {
            let arr = self.arr.iter().map(|v| -v).collect();
            Ok(Array {
                arr,
                shape: self.shape.clone(),
            })
        }
    }

    #[pyfunction(num = "50", endpoint = "false")]
    fn linspace(start: f64, end: f64, num: usize, endpoint: bool) -> Array {
        let denom: f64 = if endpoint { num - 1 } else { num } as f64;
        let get_slice = |m| start + (m as f64) / denom * (end - start);
        let arr: Vec<f64> = (0..num).map(get_slice).collect();

        Array {
            shape: vec![arr.len()],
            arr,
        }
    }

    #[pyfunction(step = "1")]
    fn arange(lim1: i64, lim2: Option<i64>, step: usize) -> Array {
        let r = if let Some(end) = lim2 {
            lim1..end
        } else {
            0..lim1
        };

        let arr: Vec<f64> = r.step_by(step).map(|n| n as f64).collect();
        let length = arr.len();
        Array {
            arr,
            shape: vec![length],
        }
    }

    #[pymodule]
    fn rnumpy(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<Array>()?;
        m.add_wrapped(wrap_pyfunction!(linspace))?;
        m.add_wrapped(wrap_pyfunction!(arange))?;
        Ok(())
    }
}
