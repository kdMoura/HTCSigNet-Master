from setuptools import setup, find_packages

setup(
    name="htcsignet",
    version="0.1.0",
    #author="Original Author",
    #author_email="author@example.com",
    description="HTCSigNet: Handwritten Signature Verification with CNN/Vision Transformer models",
    packages=find_packages(),  # automatically finds htcsignet and subpackages
    install_requires=[
        "numpy",
        "scikit-image",
        "torch",
        "torchvision",
        "einops",
        "tqdm",
        "scikit-learn",
    ],
    python_requires=">=3.8",
)
