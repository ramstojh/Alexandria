from setuptools import setup

setup(name='Alexandria-11',
      version='0.01',
      description='Collection of tools to work with Galactic Archaeology',
      url='https://github.com/ramstojh/Alexandria',
      author='Jhon Yana',
      author_email='ramstojh@alumni.usp.br',
      license='MIT',
      packages=['Alexandria-11'],
      #package_dir={'terra':'terra'},
      package_data={'Alexandria-11': ['data/*']},
      include_package_data=True,
      install_requires=['numpy', 'pandas', 'tqdm', 'astropy', 'matplotlib'],
      zip_safe=False)
