from setuptools import setup, find_packages

REQUIREMENTS = [
    'numpy >= 1.13',
    'six >= 1.10',
]

setup(
    name='oml',
    version='0.2.3',
    description='Online Machine Learning',
    author='@getumen',
    author_email='n.yoshihiro.jp@gmail.com',
    url='https://github.com/getumen/oml',
    packages=find_packages(exclude=['examples', 'tests']),
    install_requires=REQUIREMENTS,
)
