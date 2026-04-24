from setuptools import setup

package_name = "ttt_visualizer"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="ISDN3000E TA",
    maintainer_email="xingxin.he@connect.ust.hk",
    description="viser-based match visualizer for the Tic-Tac-Toe final project",
    license="MIT",
    entry_points={
        "console_scripts": [
            "ttt_visualizer_node = ttt_visualizer.visualizer_node:main",
        ],
    },
)
