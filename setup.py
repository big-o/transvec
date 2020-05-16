from setuptools import find_packages, setup


def parse_requirements(filename):
    # Copy dependencies from requirements file
    with open(filename, encoding="utf-8") as f:
        requirements = [line.strip() for line in f.read().splitlines()]
        requirements = [
            line.split("#")[0].strip()
            for line in requirements
            if not line.startswith("#")
        ]

    return requirements


def main():
    with open("README.rst", "r") as fh:
        long_description = fh.read()

    setup(
        name="transvec",
        version="0.0.2",
        description="Multilingual word embeddings.",
        long_description=long_description,
        url="http://github.com/big-o/transvec",
        author="big-o",
        author_email="big-o@users.noreply.github.com",
        license="GNU GPLv3",
        packages=find_packages(exclude="tests"),
        python_requires=">=3.7",
        install_requires=parse_requirements("requirements.txt"),
        extras_require={
            "dev": parse_requirements("requirements-dev.txt"),
            "test": parse_requirements("requirements-test.txt"),
        },
        keywords="Translation, Machine Translation, Bilingual, Multilingual, "
        "Singular Value Decomposition, SVD, Latent Semantic Indexing, "
        "LSA, LSI, Latent Dirichlet Allocation, LDA, "
        "Hierarchical Dirichlet Process, HDP, Random Projections, "
        "TFIDF, word2vec, doc2vec",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            "Topic :: Text Processing :: Linguistic",
        ],
    )


if __name__ == "__main__":
    main()
