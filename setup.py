from setuptools import setup, find_packages

setup(
    name="cecopy",
    version="0.0.1",
    author="muhammad hanif ramadhani",
    author_email="mhaniframadhani985@gmail.com",
    url="https://github.com/haniframadhani/cecopy",
    packages=find_packages(exclude=['test*', 'docs*']),
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.7"
)
