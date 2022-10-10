class Point:
    def __init__(self, e: [], expected_value):
        self.e = e
        self.expected_value = expected_value
        self.normalized_expected_value = None

    def __str__(self):
        return f"({self.e}) -> expected_value: {self.expected_value}"

    def __repr__(self):
        return f"({self.e}) -> expected_value: {self.expected_value}"


class MultiPoint:
    def __init__(self, e: [], expected_value: []):
        self.e = e
        self.expected_value = expected_value
        self.normalized_expected_value = None

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


class Node(Perceptron):
    def __init__(self, n: float, w: []):
        super().__init__(n, w)
        self.error = None

    def __str__(self):
        return f"{super().__str__()} error: {self.error}"

    def __repr__(self):
        return f"{super().__repr__()} error: {self.error}"


class Layer:
    def __init__(self, nodes: []):
        self.nodes = nodes
        self.point = Point([], 0)


class MultilayerPerceptron:
    def __init__(self, n: float, layers: []):
        self.layers = layers
        self.n = n

    def __str__(self):
        return f"n: {self.n}, w: {self.layers}"

    def __repr__(self):
        return f"n: {self.n}, w: {self.layers}"
