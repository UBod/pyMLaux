import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pyMLaux',
    version='0.0.1',
    author='Ulrich Bodenhofer',
    author_email='ulrich@bodenhofer.com',
    description='Auxiliary functions for machine learning courses and projects',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/UBod/pyMLaux',
    license='MIT',
    packages=['pyMLaux'],
    install_requires=['numpy', 'pandas', 'sklearn', 'matplotlib', 'cv2', 'math', \
                      'scipy', 'os', 'sys', 'struct', 'array'],
)
