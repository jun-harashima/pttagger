from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [ ]

test_requirements = ['numpy==1.22.0', 'torch==0.4.1', ]

setup(
    author="Jun Harashima",
    author_email='j.harashima@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Simple PyTorch-based tagger",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='pttagger',
    name='pttagger',
    packages=find_packages(include=['pttagger']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jun-harashima/pttagger',
    version='0.1.1',
    zip_safe=False,
)
