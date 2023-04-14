import setuptools

extras = {}
extras["testing"] = ["unittest"]
extras["torch"] = ["torch>=1.5.0"]
extras["torch_scatter"] = ["torch-scatter>=2.0.8"]
extras["docs"] = ["recommonmark", "sphinx", "sphinx-markdown-tables", "sphinx-rtd-theme==0.4.3", "sphinx-copybutton"]
if __name__ == "__main__":
    setuptools.setup(
        name="openprotein",
        description="Open-Protein is an open source pre-training platform that supports multiple protein pre-training models and downstream tasks.",
        version="0.2.0",
        keywords=["open", "protein", "pre-training", "downstream task"],
        license="Apache-2.0",
        package_dir={"": "src"},
        packages=setuptools.find_packages(
            where='src'
        ),
        include_package_data=True,
        url="",
        author="HIC-AI",
        author_email="hic_ai@163.com",
        install_requires=
        [
            "numpy>=1.11",
            "scikit-learn",
            "transformers",
            "scipy",
            "lmdb"
        ],
        extras_require=extras,
        python_requires=">=3.6.0,<3.10.0",
    )
