"""
HighPy: Adaptive Multi-Level Specialization Framework for Python Performance Optimization

A novel framework that addresses Python's fundamental performance bottlenecks through:
1. Multi-Level Adaptive Specialization (MLAS)
2. Type Lattice Inference with Abstract Interpretation
3. Speculative Native Loop Compilation with Guard-Based Deoptimization
4. Cross-Function Interprocedural Specialization
5. Region-Based Arena Memory Management
6. Polymorphic Inline Caching for Attribute Access
"""

from setuptools import setup, find_packages

setup(
    name="highpy",
    version="1.0.0",
    description="Adaptive Multi-Level Specialization Framework for Python Performance Optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="HighPy Research Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "cffi>=1.15.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-benchmark>=4.0",
            "matplotlib>=3.7",
            "tabulate>=0.9",
            "psutil>=5.9",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Code Generators",
    ],
)
