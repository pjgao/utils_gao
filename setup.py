import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="utils_gao",
    version="0.0.9",
    author="pipixiu",
    author_email="1783198484@qq.com",
    description="utils module for my ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pjgao",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)