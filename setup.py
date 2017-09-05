from setuptools import setup, find_packages

REQUIREMENTS = [
    'numpy >= 1.13',
    'six >= 1.10',
]

setup(
    name='oml',
    version='0.2.0',
    description='Online Machine Learning',
    author='@getumen',
    author_email='n.yoshihiro.jp@gmail.com',
    url='https://github.com/getumen/oml',
    packages=find_packages('oml'),
    install_requires=REQUIREMENTS,
    package_dir={'': 'oml'}
)
