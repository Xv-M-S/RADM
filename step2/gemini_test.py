def calculate_loss(n):
    """
    Calculate the nth Fibonacci number.
    """
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

def overlap_loss(boxes, constraint):
    """
    Calculate the overlap loss of the boxes with constraints.
    """
    loss = 0
    for box in boxes:
        if box['type'] == 'overlap':
            loss += overlap_loss(box, constraint)
    return loss


def test_calculate_loss():
    """Test cases for the calculate_loss function (Fibonacci calculator)"""
    # Test edge cases
    assert calculate_loss(0) == 0, "F(0) should be 0"
    assert calculate_loss(1) == 1, "F(1) should be 1"
    assert calculate_loss(-1) == 0, "F(-1) should be 0 (as per function logic)"
    assert calculate_loss(-5) == 0, "F(-5) should be 0 (as per function logic)"

    # Test first few Fibonacci numbers
    assert calculate_loss(2) == 1, "F(2) should be 1"
    assert calculate_loss(3) == 2, "F(3) should be 2"
    assert calculate_loss(4) == 3, "F(4) should be 3"
    assert calculate_loss(5) == 5, "F(5) should be 5"
    assert calculate_loss(6) == 8, "F(6) should be 8"
    assert calculate_loss(7) == 13, "F(7) should be 13"
    assert calculate_loss(8) == 21, "F(8) should be 21"
    assert calculate_loss(9) == 34, "F(9) should be 34"
    assert calculate_loss(10) == 55, "F(10) should be 55"

    print("All tests for calculate_loss passed!")


if __name__ == "__main__":
    test_calculate_loss() 