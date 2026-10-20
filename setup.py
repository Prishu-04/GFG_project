from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    with open(file_path) as file_obj:
        requirements = [line.strip() for line in file_obj.readlines()]

    # remove empty lines, comments and the '-e .' line
    requirements = [
        req for req in requirements
        if req and not req.startswith('#') and req != HYPEN_E_DOT
    ]

    return requirements


setup(
    name='GFG-project',
    version='0.0.1',
    author='Traffic Detection',
    author_email='prishu22genaiml@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')  # change to 'req.txt' if you keep that name
)