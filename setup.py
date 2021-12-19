from setuptools import setup

setup(name='Alexandria11',
      version='0.01',
      description='Collection of tools to work with Galactic Archaeology',
      url='https://github.com/ramstojh/Alexandria11',
      author='Jhon Yana',
      author_email='ramstojh@alumni.usp.br',
      license='MIT',
      packages=['Alexandria11'],
      #package_dir={'terra':'terra'},
      package_data={'Alexandria11': ['data/*']},
      include_package_data=True,
      install_requires=['numpy', 'pandas', 'tqdm', 'astropy', 'matplotlib'],
      zip_safe=False)
