class Point:
    def __init__(self, e, expected_value):
        self.e = e
        self.expected_value = expected_value

    def __str__(self):
        return f"({self.e}) -> expected_value: {self.expected_value}"

    def __repr__(self):
        return f"({self.e}) -> expected_value: {self.expected_value}"


class Perceptron:
    def __init__(self, n: float, w: []):
        self.w = w
        self.n = n

    def __str__(self):
        return f"n: {self.n}, w: {self.w}"

    def __repr__(self):
        return f"n: {self.n}, w: {self.w}"
