use pyo3::types::{PyFloat, PyInt, PyList};
use pyo3::{callback::IntoPyCallbackOutput, PyObjectProtocol};
use pyo3::{exceptions, prelude::*, PyNumberProtocol};
use std::collections::VecDeque;
use std::ops::{Add, Div, Mul, Sub};

pub mod rnumpy {

    use pyo3::basic::CompareOp;

    use super::*;

    #[derive(Clone, Debug)]
    enum NumericalArray {
        Int(Vec<i64>),
        Float(Vec<f64>),
    }

    impl NumericalArray {
        pub fn len(&self) -> usize {
            match &self {
                &NumericalArray::Float(arr) => arr.len(),
                &NumericalArray::Int(arr) => arr.len(),
            }
        }
    }

    impl<Target> IntoPyCallbackOutput<Target> for NumericalArray
    where
        Vec<i64>: IntoPyCallbackOutput<Target>,
        Vec<f64>: IntoPyCallbackOutput<Target>,
    {
        fn convert(self, py: Python) -> PyResult<Target> {
            match self {
                NumericalArray::Int(arr) => arr.convert(py),
                NumericalArray::Float(arr) => arr.convert(py),
            }
        }
    }

    enum Number {
        Int(i64),
        Float(f64),
    }

    impl From<Number> for i64 {
        fn from(num: Number) -> Self {
            match num {
                Number::Int(i) => i,
                Number::Float(f) => f as i64,
            }
        }
    }

    impl From<Number> for f64 {
        fn from(num: Number) -> Self {
            match num {
                Number::Int(i) => i as f64,
                Number::Float(f) => f,
            }
        }
    }

    impl<'s> FromPyObject<'s> for Number {
        fn extract(ob: &'s PyAny) -> PyResult<Self> {
            if let Ok(i) = PyInt::try_from(ob) {
                Ok(Number::Int(i.extract()?))
            } else if let Ok(f) = PyFloat::try_from(ob) {
                Ok(Number::Float(f.extract()?))
            } else {
                Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                    "{:?} cannot be converted into a number.",
                    &ob
                )))
            }
        }
    }

    #[pyclass]
    pub struct Array {
        #[pyo3(get, set)]
        pub shape: Vec<usize>,
        arr: NumericalArray,
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
            let mut elements: Option<NumericalArray> = None;
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
                    let els = &mut elements;
                    match (name, els) {
                        ("list", _) => py_lists.push_back(py_obj.downcast::<PyList>()?),
                        ("float", Some(NumericalArray::Float(ref mut arr))) => {
                            arr.push(py_obj.downcast::<PyFloat>()?.extract()?)
                        }
                        ("int", Some(NumericalArray::Int(ref mut arr))) => {
                            arr.push(py_obj.downcast::<PyInt>()?.extract()?)
                        }
                        ("int", None) => {
                            elements = Some(NumericalArray::Int(vec![py_obj
                                .downcast::<PyInt>()?
                                .extract()?]));
                        }
                        ("float", None) => {
                            elements = Some(NumericalArray::Float(vec![py_obj
                                .downcast::<PyFloat>()?
                                .extract()?]));
                        }
                        (t, els) => {
                            return Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                                "unsupported type {} for numerical array {:?}",
                                t, &els
                            )))
                        }
                    }
                }
            }
            let arr = elements.unwrap_or(NumericalArray::Float(vec![]));
            Ok(Self { arr, shape })
        }
    }

    #[pyproto]
    impl PyObjectProtocol for Array {
        fn __str__(&self) -> PyResult<String> {
            Ok(match &self.arr {
                NumericalArray::Int(arr) => format!("{:?}", arr),
                NumericalArray::Float(arr) => format!("{:?}", arr),
            })
        }

        fn __repr__(&self) -> PyResult<String> {
            Ok(format!("Array.{:?}, shape={:?}", &self.arr, &self.shape))
        }

        fn __richcmp__(
            &'p self,
            other: PyRef<Array>,
            op: pyo3::basic::CompareOp,
        ) -> PyResult<bool> {
            // TODO:
            match (&self.arr, &other.arr, &op) {
                (NumericalArray::Int(a1), NumericalArray::Int(a2), CompareOp::Eq) => Ok(a1 == a2),
                (NumericalArray::Float(a1), NumericalArray::Float(a2), CompareOp::Eq) => {
                    Ok(a1 == a2)
                }
                _ => Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Cannot perform this sort of comparison",
                )),
            }
        }

        // fn __bool__(&'p self) -> Self::Result
        // where
        //     Self: pyo3::basic::PyObjectBoolProtocol<'p>,
        // {
        //     unimplemented!()
        // }
    }

    #[pyproto]
    impl PyNumberProtocol for Array {
        fn __add__(lhs: PyRef<Array>, rhs: PyRef<Array>) -> PyResult<Array> {
            if lhs.shape != rhs.shape {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                    "arrays don't have the same shape: {:?} vs {:?}",
                    lhs.shape, rhs.shape
                )));
            }
            let arr: NumericalArray = match (&lhs.arr, &rhs.arr) {
                (NumericalArray::Int(lhs), NumericalArray::Int(rhs)) => {
                    let sum: Vec<i64> = lhs.iter().zip(rhs.iter()).map(|(l, r)| l + r).collect();
                    NumericalArray::Int(sum)
                }
                (NumericalArray::Float(lhs), NumericalArray::Float(rhs)) => {
                    let sum: Vec<f64> = lhs.iter().zip(rhs.iter()).map(|(l, r)| l + r).collect();
                    NumericalArray::Float(sum)
                }
                _ => {
                    return Err(PyErr::new::<exceptions::PyTypeError, _>(
                        "arrays don't have the same type: int vs float",
                    ));
                }
            };
            Ok(Array {
                arr,
                shape: lhs.shape.clone(),
            })
        }

        fn __sub__(lhs: PyRef<Array>, rhs: PyRef<Array>) -> PyResult<Array> {
            if lhs.shape != rhs.shape {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                    "arrays don't have the same shape: {:?} vs {:?}",
                    lhs.shape, rhs.shape
                )));
            }
            let arr: NumericalArray = match (&lhs.arr, &rhs.arr) {
                (NumericalArray::Int(lhs), NumericalArray::Int(rhs)) => {
                    let dif: Vec<i64> = lhs.iter().zip(rhs.iter()).map(|(l, r)| l - r).collect();
                    NumericalArray::Int(dif)
                }
                (NumericalArray::Float(lhs), NumericalArray::Float(rhs)) => {
                    let dif: Vec<f64> = lhs.iter().zip(rhs.iter()).map(|(l, r)| l - r).collect();
                    NumericalArray::Float(dif)
                }
                _ => {
                    return Err(PyErr::new::<exceptions::PyTypeError, _>(
                        "arrays don't have the same type: int vs float",
                    ));
                }
            };
            Ok(Array {
                arr,
                shape: lhs.shape.clone(),
            })
        }

        fn __mul__(lhs: PyRef<Array>, rhs: PyRef<Array>) -> PyResult<Array> {
            if lhs.shape != rhs.shape {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                    "arrays don't have the same shape: {:?} vs {:?}",
                    lhs.shape, rhs.shape
                )));
            }
            let arr: NumericalArray = match (&lhs.arr, &rhs.arr) {
                (NumericalArray::Int(lhs), NumericalArray::Int(rhs)) => {
                    let product: Vec<i64> =
                        lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).collect();
                    NumericalArray::Int(product)
                }
                (NumericalArray::Float(lhs), NumericalArray::Float(rhs)) => {
                    let product: Vec<f64> =
                        lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).collect();
                    NumericalArray::Float(product)
                }
                _ => {
                    return Err(PyErr::new::<exceptions::PyTypeError, _>(
                        "arrays don't have the same type: int vs float",
                    ));
                }
            };
            Ok(Array {
                arr,
                shape: lhs.shape.clone(),
            })
        }

        fn __truediv__(lhs: PyRef<Array>, rhs: PyRef<Array>) -> PyResult<Array> {
            if lhs.shape != rhs.shape {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                    "arrays don't have the same shape: {:#?} vs {:#?}",
                    lhs.shape, rhs.shape
                )));
            }
            let arr: NumericalArray = match (&lhs.arr, &rhs.arr) {
                (NumericalArray::Int(lhs), NumericalArray::Int(rhs)) => {
                    let quotient: Vec<i64> =
                        lhs.iter().zip(rhs.iter()).map(|(l, r)| l / r).collect();
                    NumericalArray::Int(quotient)
                }
                (NumericalArray::Float(lhs), NumericalArray::Float(rhs)) => {
                    let quotient: Vec<f64> =
                        lhs.iter().zip(rhs.iter()).map(|(l, r)| l / r).collect();
                    NumericalArray::Float(quotient)
                }
                _ => {
                    return Err(PyErr::new::<exceptions::PyTypeError, _>(
                        "arrays don't have the same type: int vs float",
                    ));
                }
            };
            Ok(Array {
                arr,
                shape: lhs.shape.clone(),
            })
        }
    }

    #[pyfunction(num = "50", endpoint = "false")]
    fn linspace(start: Number, end: Number, num: usize, endpoint: bool) -> Array {
        let denom: f64 = if endpoint { num - 1 } else { num } as f64;
        let start: f64 = start.into();
        let end: f64 = end.into();
        let get_slice = |m| start + (m as f64) / denom * (end - start);
        let arr: Vec<f64> = (0..num).map(get_slice).collect();

        Array {
            shape: vec![arr.len()],
            arr: NumericalArray::Float(arr),
        }
    }

    #[pyfunction(step = "1")]
    fn arange(lim1: Number, lim2: Option<Number>, step: usize) -> Array {
        let lim1: i64 = lim1.into();
        let r = if let Some(end) = lim2 {
            lim1..end.into()
        } else {
            0..lim1
        };

        let arr = NumericalArray::Int(r.step_by(step).collect());
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
