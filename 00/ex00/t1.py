from Matrix import Vector, Matrix

# Test Vector class
def test_vector():
	# Test initialization
	v1 = Vector([[1, 2, 3, 4, 5]])
	v2 = Vector([[2, 3, 4, 5, 6]])
	
	print(f"v1.data = {(v1.data)}")
	assert v1.data == [[1, 2, 3, 4, 5]]

	# Test shape property
	print(f"v1.shape = {(v1.shape)}")
	assert v1.shape == (1, 5)

	# Test addition
	print(f"v1 + v2 = {(v1 + v2)}")
	assert (v1 + v2).data == [[3, 5, 7, 9, 11]]

	# Test subtraction
	print(f"v1 - v2 = {(v1 - v2)}")
	assert (v1 - v2).data == [[-1, -1, -1, -1, -1]]

	# Test transpose
	print(f"v1.T() = {v1.T()}")
	assert (v1.T()).data == [[1], [2], [3], [4], [5]]

	# Test multiplication with a number
	print(f"v1 * 2 = {(v1 * 2)}")
	assert (v1 * 2).data == [[2, 4, 6, 8, 10]]

	# Test multiplication with another matrix
	m1 = Matrix([[1, 2, 3, 4, 5]])
	print(f"v1 * m1.T() = {(v1 * m1.T())}")
	assert (v1 * m1.T()).data == [[55]]

	# Test multiplication with another vector
	print(f"v1 * v2.T() = {(v1 * v2.T())}")
	assert (v1 * v2.T()).data == [[70]]

	# Test division
	print(f"v1 / 2 = {(v1 / 2)}")
	assert (v1 / 2).data == [[0.5, 1.0, 1.5, 2.0, 2.5]]

# Test Matrix class
def test_matrix():
    # Test initialization
    m1 = Matrix([[2, 3, 4, 5, 6]])
    v1 = Vector([[1, 2, 3, 4, 5]])
    assert m1.data == [[2, 3, 4, 5, 6]]

    # Test shape property
    assert m1.shape == (1, 5)

    # Test addition
    assert (m1 + m1).data == [[4, 6, 8, 10, 12]]

    # Test division
    assert (2 / v1).data == [[2, 1, 2/3, 1/2, 2/5]]

    # Test empty matrix
    m2 = Matrix((2, 3))
    assert m2.data == [[0, 0, 0], [0, 0, 0]]

# Test your code
def test_code():
    # Test Vector class
    test_vector()

    # Test Matrix class
    test_matrix()

# Run the tests
test_code()