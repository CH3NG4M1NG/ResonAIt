"""
ResonAIt - AGI Orchestrator Package
====================================
Package ini menyatukan semua modalitas (Teks, Gambar, Audio, Video)
menggunakan Universal Frequency Space berbasis Fourier Transform.

Instalasi: pip install resonait
"""

from setuptools import setup, find_packages
import os

# Baca README untuk long_description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Baca requirements
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="resonait",
    version="0.1.0",
    author="ResonAIt Contributors",
    author_email="resonait@example.com",
    description="AGI Orchestrator using Universal Frequency Space (Fourier-based Multimodal AI)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ResonAIt",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        # Untuk developer yang ingin menjalankan tests
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.0",
            "mypy>=1.0",
        ],
        # Untuk environment simulasi game
        "game": [
            "gymnasium>=0.29",
            "stable-baselines3>=2.0",
        ],
        # Untuk autonomous data sourcing
        "datasource": [
            "datasets>=2.0",
            "youtube-dl>=2021.12",
            "Pillow>=10.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "AGI", "LLM", "Fourier", "multimodal", "frequency",
        "neural-operator", "AI", "deep-learning", "pytorch"
    ],
    entry_points={
        "console_scripts": [
            # CLI tool untuk menjalankan alignment dari terminal
            "resonait-align=resonait.tools.alignment:main_cli",
            # CLI tool untuk menjalankan environment hook
            "resonait-env=resonait.environment.hook:main_cli",
        ],
    },
    # Memungkinkan komunitas mendaftarkan modul sensor baru sebagai plugin
    # Contoh: pip install resonait-sensor-lidar
    # akan otomatis terdaftar di resonait.sensors
    entry_points={
        "resonait.sensors": [
            # Plugin sensor bawaan
            "text = resonait.converters.text_converter:TextConverter",
            "image = resonait.converters.image_converter:ImageConverter",
            "audio = resonait.converters.audio_converter:AudioConverter",
        ],
        "console_scripts": [
            "resonait-align=resonait.tools.alignment:main_cli",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ResonAIt/issues",
        "Documentation": "https://resonait.readthedocs.io",
        "Source": "https://github.com/yourusername/ResonAIt",
    },
)
