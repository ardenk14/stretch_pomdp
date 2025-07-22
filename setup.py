from setuptools import find_packages, setup

package_name = 'stretch_pomdp'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ardenk14',
    maintainer_email='ardenk14@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pomdp_manager = stretch_pomdp.pomdp_manager:main',
            'action_follower = stretch_pomdp.action_follower:main',
        ],
    },
)
