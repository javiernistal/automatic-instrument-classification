from distutils.core import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session='hack')
setup(
    name='EstimatorSelection',
    version='1.0.0',
    author='J. Nistal Hurle',
    author_email='j.nistalhurle@gmail.com',
    packages=['instrument_classification'],
    scripts=['instrument_classification/gs_params.py', 'instrument_classification/estimator_sel/estimator_selection.py'],
    url='http://pypi.python.org/pypi/EstimatorSelection/',
    license='LICENSE.txt',
    description='Estimator and dimensionality reduction method selection',
    long_description=open('README.md').read(),
    install_requires=[
        str(ir.req) for ir in install_reqs
    ],
)
