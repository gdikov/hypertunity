import argparse
import os
import pickle
import sys


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=str)
    parser.add_argument("output_file", type=str)
    return parser.parse_args(args)


def main(x: int, y: float, z: str) -> float:
    if z.endswith(tuple("0123456789")):
        return y * x
    return y * x ** 2


if __name__ == '__main__':
    parsed_args = parse_args(sys.argv[1:])
    result = main(parsed_args.x, parsed_args.y, parsed_args.z)
    print(result)
    output_dir = os.path.dirname(parsed_args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(parsed_args.output_file, 'wb') as fp:
        pickle.dump(result, fp)
