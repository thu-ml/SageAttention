"""
Copyright 2024 SageAttention team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from setuptools import setup, find_packages

setup(
    name='sageattention', 
    version='1.0.4',  
    author='Jintao Zhang, Haofeng Huang',
    author_email='jt-zhang6@gmail.com, huanghf22@mails.tsinghua.edu.cn', 
    packages=find_packages(),  
    description='Accurate and efficient quantized plug-and-play attention.',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown', 
    url='https://github.com/thu-ml/SageAttention', 
    license='Apache 2.0 License',  
    python_requires='>=3.9', 
    classifiers=[  
        'Development Status :: 4 - Beta', 
        'Intended Audience :: Developers',  
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: Apache 2.0 License',  
        'Programming Language :: Python :: 3', 
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
)
