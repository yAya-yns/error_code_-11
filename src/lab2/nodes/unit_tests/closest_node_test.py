import numpy as np

def closest_node(nodes, point : np.ndarray) -> int:
    '''
    Returns the index of the closest node
    '''
    assert point.shape == (2, 1)

    nodes = np.hstack(nodes)
    print(nodes.shape)
    diff = nodes - point
    dist = np.sum(np.square(diff), axis=0)
    return np.argmin(dist)

if __name__ == '__main__':
    nodes = [
        np.array([2, 1]).reshape(2, 1),
        np.array([1, 0]).reshape(2, 1),
        np.array([3, 4]).reshape(2, 1)
    ]

    point = np.array([3, 1]).reshape(2, 1)

    print(closest_node(nodes, point))