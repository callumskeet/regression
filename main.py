import argparse
import tests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-test", action="store_true")
    parser.add_argument("--logistic-test", action="store_true")
    args = parser.parse_args()

    if args.linear_test:
        tests.linear_test()
    elif args.logistic_test:
        tests.logistic_test()


if __name__ == "__main__":
    main()
