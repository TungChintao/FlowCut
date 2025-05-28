from setuptools import setup, find_packages

setup(
    name="flowcut",
    version="1.0.2",
    packages=find_packages(),
    author="Jintao Tong",
    author_email="jintaotong@hust.edu.cn",
    description="FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TungChintao/FlowCut",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)