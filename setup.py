from distutils.core import setup

setup(name='OWE',
      version='0.1a',
      description='Open World Extension for Knowledge Graph Completion Models',
      author='Haseeb Shah and Johannes Villmow',
      author_email='hshah1@ualberta.ca',
      license='gnu',
      packages=['owe'],
      url="https://www.aclweb.org/anthology/2020.textgraphs-1.9/", 
      install_requires=[
        'torch==1.0.0',
        'numpy==1.16.0',
        'gensim==3.7.0',
        'tabulate==0.8.2',
        'matplotlib==3.0.2',
        'nltk==3.4',
        'annoy==1.15.0',
        'tensorflow',
        'tensorboardX==1.6',
        'tqdm==4.30.0',
        'attrs',
      ],
      # package_data={'owe': ['data/*.pickle', 'data/*.gz']}
     )
