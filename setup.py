import os

from setuptools import find_packages, setup

setup(
    name='joernti',
    version=os.popen("git describe --tags --abbrev=0").read().lstrip("v").rstrip(),
    description="A collection of type inference models built around information extracted from Joern-generated code "
                "property graphs.",
    author='David Baker Effendi, Lukas Seidel, Xavier Pinho',
    license='MIT',
    packages=find_packages(
        include=['joernti', 'joernti.util', 'joernti.domain', 'joernti.llm']
    ),
    install_requires=['numpy', 'pandas', 'nptyping', 'typer', 'torch', 'optimum[onnxruntime]'],
    use_scm_version=True,
    setup_requires=['pytest-runner', 'setuptools_scm'],
    entry_points={'console_scripts': ['joernti=joernti.__main__:main']},
    tests_require=['pytest==7.1.3'],
    test_suite='tests',
)
