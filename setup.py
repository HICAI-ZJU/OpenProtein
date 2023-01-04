import setuptools

extras = {}
extras["testing"] = ["unittest"]
extras["torch"] = ["torch>=1.5.0"]
extras["docs"] = ["recommonmark", "sphinx", "sphinx-markdown-tables", "sphinx-rtd-theme==0.4.3", "sphinx-copybutton"]
if __name__ == "__main__":
    setuptools.setup(
        name="openprotein",
        description="",
        version="0.0.1",
        keywords=["open", "protein", "pre-training", "downstream task"],
        license="Apache-2.0",
        package_dir={"": "src"},
        packages=setuptools.find_packages(
            where='src/openprotein'
        ),
        include_package_data=True,

        url="",
        author="HIC-AI",
        author_email="hic_ai@163.com",

        install_requires=
        [
            "numpy>=1.11",
            "scikit-learn",
            "scipy",
            "lmdb"
        ],
        extras_require=extras,
        python_requires=">=3.6.0,<=3.10.0",
    )
