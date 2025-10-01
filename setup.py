from setuptools import setup, find_packages

# reqs = [str(ir.req) for ir in pip.req.parse_requirements(
#     'requirements.txt',
#     session=pip.download.PipSession())]

setup(
    name='probabilistic_robotics_python_examples',
    version='0.0.1',
    author='Mauro Martini',
    author_email='mauro.martini@polito.it',
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open('requirements.txt')] + [],
    dependency_links=[
        " "
    ]
)
