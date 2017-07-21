from setuptools import setup, find_packages

REQUIREMENTS = [
    'numpy >= 1.13',
]

setup(
    name='oml',
    version='0.1.0',
    description='Online Machine Learning',
    author='@getumen',
    author_email='yoshihiro[AT]mdl.cs.tsukuba.ac.jp',
    url='https://github.com/getumen/oml',
    packages=find_packages(exclude=('tests', )),
    install_requires=REQUIREMENTS,
)
