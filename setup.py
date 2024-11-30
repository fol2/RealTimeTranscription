from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the README for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Ensure the config directory exists and is included
os.makedirs('config', exist_ok=True)

setup(
    name='realtime-transcriber',
    version='1.0.0',
    description='Real-time audio transcription system with configurable pipeline',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/realtime-transcriber',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    
    # Include package data
    package_data={
        '': [
            'config.ini',
            'config/*.json',
            'models/.gitkeep',
            'logs/.gitkeep'
        ],
    },
    
    # Create required directories
    data_files=[
        ('config', ['config.ini']),
        ('logs', []),
        ('models', []),
    ],
    
    # Entry points for command-line usage
    entry_points={
        'console_scripts': [
            'transcribe=src.main:main',
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent',
    ],
    
    # Additional metadata
    keywords='speech recognition, transcription, real-time, whisper, audio processing',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/realtime-transcriber/issues',
        'Source': 'https://github.com/yourusername/realtime-transcriber',
    },
) 