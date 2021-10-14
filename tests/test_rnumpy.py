import rnumpy as rnp
import numpy as np


def test_init_float():
    rnp.Array([1.0, 2.0, 3.0])


def test_init_int():
    rnp.Array([1])


def test_multiple_dim():
    x = rnp.Array([[1.0, 2.0], [3.0, 4.0]])
    print(x.shape)
    assert x.shape == [2, 2]


def test_index():
    x = rnp.Array([[10.0]])
    assert x[0, 0] == 10


def test_add():
    x = rnp.Array([[1.0, 2.0], [3.0, 4.0]])
    y = rnp.Array([[1.0, 2.0], [3.0, 4.0]])
    res = rnp.Array([[2.0, 4.0], [6.0, 8.0]])
    assert x + y == res


def test_sub():
    x = rnp.Array([[1.0, 2.0], [3.0, 4.0]])
    y = rnp.Array([[2.0, 1.0], [3.0, -4.0]])
    res = rnp.Array([[-1.0, 1.0], [0.0, 8.0]])
    assert x - y == res


def test_linspace():
    x = rnp.linspace(1.0, 10.0, 10, endpoint=True)
    print("hi")
    assert x.arr == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def test_arange():
    single_number = rnp.arange(3)
    val_range = rnp.arange(1, 3)
    step = rnp.arange(3, 7, 2)

    assert single_number.arr == [0.0, 1.0, 2.0]
    assert val_range.arr == [1.0, 2.0]
    assert step.arr == [3.0, 5.0]


def test_reshape():
    x = rnp.Array([1.0, 2.0, 3.0, 4.0]).reshape((2, 2))
    assert x.shape == [2, 2]
