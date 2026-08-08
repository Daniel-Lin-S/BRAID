import setuptools

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BRAID",
    version="1.0.0",
    author="Parsa Vahidi, Omid Sani",
    author_email="pvahidi@usc.edu, omidsani@gmail.com",
    description=(
        "Python implementation for BRAID "
        "(Behaviorally Relevant Analysis of Intrinsic Dynamics)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShanechiLab/BRAID",
    packages=setuptools.find_packages(),
    install_requires=["PyYAML==6.0.2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
