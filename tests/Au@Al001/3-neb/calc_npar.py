#! /usr/bin/env python
"""
Calculate the number of adjustable parameters of neural network according to
its topology.
"""

import sys


def calc_size_descriptor(nmax):
    """Calculate the size of Zernike-type descriptor."""
    count = 0
    for n in range(nmax+1):
        for l in range(n+1):
            if (n-l) % 2 == 0:
                count += 1
    return count


def calc_num_parameter(nmax, hidden_layers):
    """Calculate the number of adjustable parameters."""
    size_descriptor = calc_size_descriptor(nmax)
    topology = [size_descriptor]
    topology.extend(hidden_layers)
    topology.append(1)
    
    count = 0
    for i in range(len(topology)-1):
        count += (topology[i]+1) * topology[i+1]
    return count


def main():
    nmax = int(sys.argv[1])
    hidden_layers = [int(i) for i in sys.argv[2:]]
    print(calc_size_descriptor(nmax))
    print(calc_num_parameter(nmax, hidden_layers))


if __name__ == "__main__":
    main()
