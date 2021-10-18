use pyo3::basic::CompareOp;
use pyo3::types::{PyFloat, PyInt, PyList};
use pyo3::{callback::IntoPyCallbackOutput, PyMappingProtocol, PyObjectProtocol};
use pyo3::{exceptions, prelude::*, PyNumberProtocol};
use std::collections::VecDeque;
use std::ops::Range;

pub mod rnumpy {

    use super::*;

    #[derive(Clone, Debug, FromPyObject)]

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

        fn get(&self, index: usize) -> Number {
            match &self {
                NumericalArray::Float(f) => Number::Float(f[index]),
                NumericalArray::Int(i) => Number::Int(i[index]),
            }
        }

        fn get_range(&self, r: Range<usize>) -> NumericalArray {
            match &self {
                NumericalArray::Float(f) => NumericalArray::Float(f[r].to_vec()),
                NumericalArray::Int(i) => NumericalArray::Int(i[r].to_vec()),
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

    #[derive(FromPyObject, Clone, Copy)]
    pub enum Number {
        Int(i64),
        Float(f64),
    }

    impl<T> IntoPyCallbackOutput<T> for Number
    where
        f64: IntoPyCallbackOutput<T>,
        i64: IntoPyCallbackOutput<T>,
    {
        fn convert(self, py: Python) -> PyResult<T> {
            match self {
                Number::Int(i) => i.convert(py),
                Number::Float(f) => f.convert(py),
            }
        }
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

    #[pyclass]
    pub struct Array {
        #[pyo3(get, set)]
        pub shape: Vec<usize>,
        arr: NumericalArray,
    }

    #[pymethods]
    impl Array {
        #[new]
        pub fn py_new(arr: Self) -> Self {
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
            let arr_str = match &self.arr {
                NumericalArray::Int(arr) => format!("{:?}", arr),
                NumericalArray::Float(arr) => format!("{:?}", arr),
            };
            Ok(format!("Array({})", arr_str))
        }

        fn __richcmp__(&'p self, other: PyRef<Array>, op: pyo3::basic::CompareOp) -> bool {
            let arrays_match = match (&self.arr, &other.arr, &op) {
                (NumericalArray::Int(a1), NumericalArray::Int(a2), CompareOp::Eq) => a1 == a2,
                (NumericalArray::Float(a1), NumericalArray::Float(a2), CompareOp::Eq) => a1 == a2,

                _ => false,
            };
            let shapes_match = self.shape == other.shape;
            arrays_match && shapes_match
        }
    }

    #[derive(FromPyObject)]
    pub enum ArrayIndexInput {
        Array(Vec<isize>),
        Number(isize),
    }
    pub enum ArrayIndexOutput {
        Array(Array),
        Number(Number),
    }

    // impl From<isize> for ArrayIndexInput {
    //     fn from(num: isize) -> Self {
    //         ArrayIndexInput::Number(num)
    //     }
    // }

    impl IntoPy<PyObject> for ArrayIndexOutput {
        fn into_py(self, py: Python) -> PyObject {
            match self {
                ArrayIndexOutput::Array(arr) => arr.convert(py).unwrap(),
                ArrayIndexOutput::Number(num) => num.convert(py).unwrap(),
            }
        }
    }

    #[pyproto]
    impl PyMappingProtocol for Array {
        fn __len__(&self) -> usize {
            return self.shape[0];
        }
        fn __getitem__(&self, idx: ArrayIndexInput) -> PyResult<ArrayIndexOutput> {
            //TODO: Finish new input logic
            let indices = match idx {
                ArrayIndexInput::Number(i) => vec![i],
                ArrayIndexInput::Array(arr) => arr,
            };
            // TODO negative indexing
            let indices: Vec<usize> = indices
                .iter()
                .map(|idx| if *idx > 0 { *idx as usize } else { 0 })
                .collect();
            let mut idx = 0;
            let mut stride = self.shape.iter().fold(1, |stride, dim| stride * dim);
            let output_is_array = indices.len() < self.shape.len();
            for (index, dim) in indices.iter().zip(self.shape.iter()) {
                stride /= dim;
                idx += index * stride;
            }
            if output_is_array {
                let start = idx;
                let end = idx
                    + self.shape[indices.len()..]
                        .iter()
                        .fold(1, |stride, dim| stride * dim);

                let subarray = self.arr.get_range(start..end);
                let shape: Vec<usize> = self.shape[indices.len()..].to_vec();
                Ok(ArrayIndexOutput::Array(Array {
                    arr: subarray,
                    shape,
                }))
            } else {
                let num: Number = self.arr.get(idx);
                Ok(ArrayIndexOutput::Number(num))
            }
        }
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
                    let dif: Vec<i64> = lhs.iter().zip(rhs.iter()).map(|(l, r)| l + r).collect();
                    NumericalArray::Int(dif)
                }
                (NumericalArray::Float(lhs), NumericalArray::Float(rhs)) => {
                    let dif: Vec<f64> = lhs.iter().zip(rhs.iter()).map(|(l, r)| l + r).collect();
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
