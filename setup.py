from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read()

setup(
    name='int2vec',
    version='0.0.1',
    description='Distributed representations for integers',
    long_description=readme,
    author='Dan Busbridge',
    author_email='dan.busbridge@babylonhealth.com',
    url='https://github.com/dbusbridge/int2vec',
    install_requires=requirements,
    packages=find_packages(exclude=('tests', 'docs'))
)

