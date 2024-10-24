from setuptools import setup, find_packages

setup(
    name='sageattention', 
    version='1.0.3',  
    author='Jintao Zhang',  
    author_email='jt-zhang6@gmail.com', 
    packages=find_packages(),  
    description='Accurate and efficient 8-bit plug-and-play attention.',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown', 
    url='https://github.com/jt-zhang/SageAttention', 
    license='MIT',  
    python_requires='>=3.9', 
    classifiers=[  
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Developers',  
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3', 
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
)
