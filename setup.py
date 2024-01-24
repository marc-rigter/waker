import setuptools
import pathlib


setuptools.setup(
    name='waker',
    version='1.0.0',
    description='Reward Free Curricula for Training Robust World Models',
    url='',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['waker'],
    package_data={'waker': ['configs.yaml']},
    entry_points={'console_scripts': ['waker=waker.train:main']},
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
