from glob import glob
from setuptools import setup

package_name = "ttt_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="ISDN3000E TA",
    maintainer_email="xingxin.he@connect.ust.hk",
    description="Launch files for the Tic-Tac-Toe final project stack",
    license="MIT",
    entry_points={
        "console_scripts": [
            "ik_pose_audit = ttt_bringup.ik_pose_audit:main",
        ],
    },
)
