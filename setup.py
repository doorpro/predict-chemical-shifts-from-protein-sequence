from setuptools import setup

setup(
    name='my_script_package',
    version='0.1',
    description='A simple script package.',
    author='Your Name',
    author_email='your.email@example.com',
    url='http://example.com',
    packages=['my_script'],
    scripts=['my_script.py'],
    entry_points={
        'console_scripts': [
            'py.file=my_script:main',
        ],
    },
)