import numpy as np


def fair_max(elements: list | np.ndarray, key = lambda x: x) -> int:
    """
    Returns the index of the maximum element in the collection <elements> by randomly resolving ties.
    """
    max_value = key(max(elements, key=key))
    max_elements = [x for x in elements if key(x) == max_value]
    return max_elements[np.random.choice(len(max_elements))]


def main():

    # test fair_max (at least it is a max operator)
    elements = [1, 2, 3, 3, 3, -4, -5]
    assert fair_max(elements) == 3


if __name__ == "__main__":
    main()