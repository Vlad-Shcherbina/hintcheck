from setuptools import setup

setup(
    name='hintcheck',
    description='Runtime type checks for PEP 484 annotations',
    author='Vlad Shcherbina',
    author_email='vlad.shcherbina@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Testing',
    ],
    keywords='gradual typing 484',
    py_modules=['hintcheck'],
)
