import numpy as np

import socket

hostname = socket.gethostname()


def main(args):
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="chi3d")
    args = parser.parse_args()

    main(args)
