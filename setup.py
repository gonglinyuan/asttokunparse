#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re

from setuptools import setup, find_packages


def read_reqs(name):
    with open(os.path.join(os.path.dirname(__file__), name)) as f:
        return [line for line in f.read().split('\n') if line and not line.strip().startswith('#')]


tests_require = []  # mostly handled by tox


def read_version():
    with open(os.path.join('lib', 'asttokunparse', '__init__.py')) as f:
        m = re.search(r'''__version__\s*=\s*['"]([^'"]*)['"]''', f.read())
        if m:
            return m.group(1)
        raise ValueError("couldn't find version")


setup(
    name='asttokunparse',
    version=read_version(),
    description='An AST unparser for Python with tokenization',
    maintainer='Linyuan Gong',
    maintainer_email='gonglinyuan@hotmail.com',
    url='https://github.com/gonglinyuan/asttokunparse',
    packages=find_packages('lib'),
    package_dir={'': 'lib'},
    include_package_data=True,
    install_requires=read_reqs('requirements.txt'),
    license="BSD",
    zip_safe=False,
    keywords='asttokunparse',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Code Generators',
    ],
    test_suite='tests',
    tests_require=tests_require,
)
