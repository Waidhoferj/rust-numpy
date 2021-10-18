import rnumpy as rnp
import numpy as np


def test_init_float():
    rnp.Array([1.0, 2.0, 3.0])


def test_init_int():
    rnp.Array([1])


def test_comparison():
    # Same arrays should match
    x = rnp.Array([1, 2, 3, 4])
    y = rnp.Array([1, 2, 3, 4])
    assert x == y
    # But arrays of different types shouldn't
    z = rnp.Array([1.0, 2.0, 3.0, 4.0])
    assert not (x == z)
    # multi dim arrays should match
    x = x.reshape([2, 2])
    y = y.reshape([2, 2])
    assert x == y
    # but not if they have different shapes
    y = y.reshape([4])
    assert not (x == y)


def test_multiple_dim():
    x = rnp.Array([[1.0, 2.0], [3.0, 4.0]])
    assert x.shape == [2, 2]


def test_simple_index():
    x = rnp.Array([10])
    assert x[0] == 10


def test_multi_dim_index():
    x = rnp.Array([[10]])
    assert x[[0, 0]] == 10
    assert x[0, 0] == 10
    x = rnp.arange(10).reshape([2, 5])
    assert x[0] == rnp.arange(5)
    assert x[1] == rnp.arange(5, 10)


def test_add():
    x = rnp.Array([[1.0, 2.0], [3.0, 4.0]])
    y = rnp.Array([[1.0, 2.0], [3.0, 4.0]])
    res = rnp.Array([[2.0, 4.0], [6.0, 8.0]])
    assert x + y == res


def test_subtract():
    x = rnp.Array([[1, 2], [3, 4]])
    y = rnp.Array([[2, 1], [3, -4]])
    res = rnp.Array([[-1, 1], [0, 8]])
    assert x - y == res


def test_multiply():
    x = rnp.Array([[1, 2], [3, 4]])
    y = rnp.Array([[2, 1], [3, 4]])
    res = rnp.Array([[2, 2], [9, 16]])
    assert x * y == res


def test_divide():
    x = rnp.Array([[1, 2], [3, 4]])
    y = rnp.Array([[2, 1], [3, 4]])
    res = rnp.Array([[0, 2], [1, 1]])
    assert x / y == res


def test_linspace():
    x = rnp.linspace(1.0, 10.0, 10, endpoint=True)
    assert x == rnp.Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


def test_arange():
    single_number = rnp.arange(3)
    val_range = rnp.arange(1, 3)
    step = rnp.arange(3, 7, 2)

    assert single_number == rnp.Array([0, 1, 2])
    assert val_range == rnp.Array([1, 2])
    assert step == rnp.Array([3, 5])


def test_reshape():
    x = rnp.Array([1.0, 2.0, 3.0, 4.0]).reshape((2, 2))
    assert x.shape == [2, 2]
