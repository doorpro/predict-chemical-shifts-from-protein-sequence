from setuptools import setup
import os 

setup(
    name='plm-cs',
    version='0.3',
    description='Protein chemical shift prediction based on Protein Language Model',
    author='Zhu He',
    author_email='2260913071@qq.com',
    url='https://github.com/doorpro/predict-chemical-shifts-from-protein-sequence.git',
    packages=["plm-cs"],
    long_description = open('README.md',encoding="utf-8").read(),
    zip_safe=True,
    include_package_data=True,
    # package_data={
    #     "plm-cs":["esm_ckpt/esm2_t33_650M_UR50D.pt",
    #               "ckpt/*"]
    # },
    entry_points = {
        'console_scripts': [
            'plm-cs = CS_predict:main'
        ]
    },
    install_requires=[
        'pandas',
        'torch >= 2.0.0',
        'esm >= 3.0.3',
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)