from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ArtRAGSys",
    version="0.1.0",
    author="Bohdan Zhvalevksyi",
    author_email="zhvalevskyiph@gmail.com",
    description="ArtRAG System: Art Retrieval-Augmented Generation System with Modern GUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas",
        "chromadb",
        "sentence-transformers",
        "spacy",
        "textblob",
        "rapidfuzz",
        "nltk",
        "requests",
        "customtkinter",
        "Pillow",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": ["artrag-gui = gui_modern:main"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
