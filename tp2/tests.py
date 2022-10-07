import numpy

def tests():
    inner_layers = 2
    nodes_count = 2
    dim = 2
    output_nodes = 1


    weights = {}
    for j in range(inner_layers):
        weights[j] = []
        for i in range(nodes_count):
            weights[j].append(numpy.random.uniform(-1, 1, size=(dim + 1)))

    weights[inner_layers] = []
    for i in range(output_nodes):
        weights[inner_layers].append(numpy.random.uniform(-1, 1, size=(dim + 1)))


    arr1 = [1, 2, 3]
    value = [1]
    print(arr1[1:])
    print(arr1)



    print(weights)

if __name__ == "__main__":
    tests()
