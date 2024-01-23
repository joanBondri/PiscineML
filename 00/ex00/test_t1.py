from Matrix import Vector, Matrix

# Test Vector class
def test_vector():
    # Test initialization
    v1 = Vector([1, 2, 3, 4, 5])
    assert v1.data == [1, 2, 3, 4, 5]

    # Test shape property
    assert v1.shape == (5,)

    # Test addition
    v2 = Vector([2, 3, 4, 5, 6])
    assert v1 + v2 == Vector([3, 5, 7, 9, 11])

    # Test subtraction
    assert v1 - v2 == Vector([-1, -1, -1, -1, -1])

    # Test transpose
    assert v1.T() == Matrix([[1], [2], [3], [4], [5]])

# Test Matrix class
def test_matrix():
    # Test initialization
    m1 = Matrix([[2, 3, 4, 5, 6]])
    assert m1.data == [[2, 3, 4, 5, 6]]

    # Test shape property
    assert m1.shape == (1, 5)

    # Test addition
    assert m1 + m1 == Matrix([[4, 6, 8, 10, 12]])

    # Test division
    assert 2 / v1 == Vector([2, 1, 2/3, 1/2, 2/5])

    # Test empty matrix
    m2 = Matrix((2, 3))
    assert m2.data == [[0, 0, 0], [0, 0, 0]]

# Run the tests
test_vector()
test_matrix()