# Rust NumPy

What if [NumPy]() was written in Rust? Here's a proof of concept implementation.

## Getting Started

### Prerequisites

- [Rust/Cargo](https://www.rust-lang.org/tools/install)
- [Python](https://www.python.org/downloads/)

### Setup

Get Rust to Python compiler

```bash
pip install maturin
```

Compile `rnumpy` library

```bash
maturin develop
```

Try it out:

```bash
python
>>> import rnumpy as rnp
>>> rnp.arange(3)
Array.Int([0, 1, 2]), shape=[3
```

### Tests

All of the functionality of the library is outlined in the `tests` folder. To run the tests, you'll need to `pip install pytest`. Then run the following commands to compile the library and run the tests:

```
pip install maturin
pytest
```

## Resources

- [PyO3](https://github.com/PyO3/pyo3): Rust <-> Python bindings
