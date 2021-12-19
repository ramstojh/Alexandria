from setuptools import setup

setup(name='Alexandria',
      version='0.01',
      description='Script to simulate planet formation or engulfment',
      url='https://github.com/ramstojh/Alexandria',
      author='Jhon Yana',
      author_email='ramstojh@alumni.usp.br',
      license='MIT',
      packages=['Alexandria'],
      #package_dir={'terra':'terra'},
      package_data={'Alexandria': ['data/*']},
      include_package_data=True,
      install_requires=['numpy', 'pandas', 'tqdm', 'astropy', 'matplotlib'],
      zip_safe=False)
