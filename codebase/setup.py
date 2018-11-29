from distutils.core import setup
import os


install_reqs = [ 'pynn==0.7.5', 'numpy', 'scipy', 'matplotlib']
pack = 'spikevo'
pack_dir = os.path.join(os.path.dirname(__file__), pack)

setup(name='SpikEvo',
      version='0.0.1',
      description='Evolutionary experiments with spiking neural networks',
      author='Garibaldi Pineda Garcia',
      author_email='g.pineda-garcia@sussex.ac.uk',
      url='https://github.com/chanokin/brainscales-recognition',
      install_requires=install_reqs,
     )
