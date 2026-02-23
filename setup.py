from setuptools import setup, find_packages

NAME = "vjepa2"
VERSION = "0.0.1"
DESCRIPTION = "PyTorch code and models for V-JEPA 2."
URL = "https://github.com/facebookresearch/vjepa2"


def get_requirements():
    with open("./requirements.txt") as reqsf:
        return reqsf.read().splitlines()


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        url=URL,
        python_requires=">=3.11",
        packages=find_packages(include=["app", "app.*", "src", "src.*", "third_party", "third_party.*"]),
        install_requires=get_requirements(),
    )
