from setuptools import setup, find_packages

setup(
    name="pyheifconcat",
    version="0.1.0",
    packages=find_packages(exclude=["test_*"]),
    url="https://github.com/satyaog/pyheifconcat",
    license="The MIT License",
    author="Satya Ortiz-Gagne",
    author_email="satya.ortiz-gagne@mila.quebec",
    description="",
    requires=["pybzparse"],
    install_requires=["pybzparse"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    long_description=""
)
