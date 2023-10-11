from setuptools import setup

setup(
    name="your_code",
    version="0.1",
    description="Word2Vec and Phrase Similarity Calculator",
    packages=["your_code"],
    install_requires=["gensim", "numpy", "scikit-learn", "pandas"],
    entry_points={"console_scripts": ["your_code = your_code.cli:main"]},
)
