from setuptools import find_packages, setup


def parse_requirements(filename):
    # Copy dependencies from requirements file
    with open(filename, encoding='utf-8') as f:
        requirements = [line.strip() for line in f.read().splitlines()]
        requirements = [line.split('#')[0].strip() for line in requirements
                        if not line.startswith('#')]

    return requirements


def main():
    with open("README.rst", "r") as fh:
        long_description = fh.read()

    setup(
        name="transvec",
        version="0.0.1",
        description="Multilingual word embeddings.",
        long_description=long_description,
        url="http://github.com/big-o/transvec",
        author="big-o",
        author_email="big-o@github",
        license="GNU GPLv3",
        packages=find_packages(exclude="tests"),
        python_requires=">=3.7",
        install_requires=parse_requirements('requirements.txt'),
        extras_require={
            'dev': parse_requirements('requirements-dev.txt'),
            'test': parse_requirements('requirements-test.txt')
        },
    )


if __name__ == "__main__":
    main()
