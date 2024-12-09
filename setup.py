import setuptools

setuptools.setup(
    name="pepinvent",
    version="0.0.1",
    author="Gokce Geylan",
    author_email="gokce.geylan@astrazeneca.com",
    description="peptide generative model framework coupled with reinforcement learning for peptide design",
    long_description_content_type="text/markdown",
    url="/github/usr/url",
    packages=setuptools.find_packages(exclude='unit_tests'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)