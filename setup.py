from setuptools import setup

setup(
    name='dd',
    version='0.1',
    description='tools for debris disk modelling',
    url='http://github.com/drgmk/dd',
    author='Grant M. Kennedy',
    author_email='g.kennedy@warwick.ac.uk',
    license='MIT',
    packages=['dd'],
    classifiers=['Programming Language :: Python :: 3'],
    install_requires = ['numpy','scipy'],
    zip_safe=False
    )
