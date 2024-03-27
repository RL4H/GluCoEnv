from setuptools import setup

setup(
    name='glucoenv',
    version='0.1.0',
    packages=['glucoenv', 'glucoenv.env', 'glucoenv.agent', 'glucoenv.visualiser'],
    install_requires=[
            'PyYAML==6.0.1',
            'pandas==1.3.4',
            'numpy==1.26.3',
            'matplotlib==3.5.0',
            'torch==2.2.0+cu118',
            'torchaudio==2.2.0+cu118',
            'torchvision==0.17.0+cu118',
            'torchdiffeq==0.2.3',
            'torchcubicspline @ git+https://github.com/patrick-kidger/torchcubicspline.git',
        ],
    url='capsml.com',
    license='MIT License',
    author='Chirath Hettiarachchi',
    author_email='chirathyh@gmail.com',
    description='Glucose Control Environment'
)
