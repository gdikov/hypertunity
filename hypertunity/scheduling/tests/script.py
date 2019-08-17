import argparse
import sys


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=str, required=True)
    return parser.parse_args(args)


def main(x: int, y: float, z: str) -> float:
    if z.endswith(tuple("0123456789")):
        return y * x
    return y * x**2


if __name__ == '__main__':
    parsed_args = parse_args(sys.argv[1:])
    result = main(parsed_args.x, parsed_args.y, parsed_args.z)
    print(result)
