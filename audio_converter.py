"""
ResonAIt - AGI Orchestrator Package
=====================================

ResonAIt adalah framework AGI yang menggunakan Universal Frequency Space
sebagai bahasa universal untuk menyatukan semua modalitas persepsi AI.

Arsitektur Utama:
    - Universal Frequency Converter : Mengubah semua input ke domain frekuensi
    - AGI Core (FNO Brain)          : Memproses informasi di ruang frekuensi
    - Alignment Tool                : Meng-upgrade LLM privat ke format ResonAIt
    - Pain System                   : Simulasi rasa sakit via interferensi frekuensi
    - Environment Hook              : Koneksi ke mesin game untuk belajar mandiri

Contoh Penggunaan Cepat:
    >>> from resonait import ResonAItBrain
    >>> brain = ResonAItBrain.from_config("config.yaml")
    >>> freq = brain.perceive(text="Halo dunia!", modality="text")
    >>> output = brain.think(freq)

Author: ResonAIt Contributors
License: MIT
"""

__version__ = "0.1.0"
__author__ = "ResonAIt Contributors"

# === IMPOR UTAMA (Lazy Loading untuk efisiensi startup) ===
# Komponen-komponen ini hanya di-load saat dibutuhkan

def _lazy_import():
    """Lazy loader untuk menghindari circular imports dan mempercepat startup."""
    pass

# Ekspor utama yang langsung tersedia saat 'import resonait'
from resonait.core.brain import ResonAItBrain
from resonait.core.frequency_space import FrequencyTensor, UniversalFrequencySpace
from resonait.converters.universal_converter import UniversalFrequencyConverter
from resonait.pain.dissonance import DissonanceEngine

# Plugin Registry - komunitas bisa mendaftarkan sensor baru di sini
# Format: {"nama_sensor": KelasConverter}
SENSOR_REGISTRY: dict = {}

def register_sensor(name: str, converter_class):
    """
    Daftarkan sensor/konverter baru ke ResonAIt.
    
    Ini adalah extension point utama untuk komunitas GitHub.
    Setiap modul sensor baru cukup memanggil fungsi ini untuk
    terintegrasi penuh dengan ekosistem ResonAIt.
    
    Args:
        name: Nama unik sensor (e.g., "lidar", "eeg", "thermal")
        converter_class: Kelas yang mengimplementasikan BaseConverter
    
    Contoh:
        >>> from resonait import register_sensor
        >>> from my_plugin import LidarConverter
        >>> register_sensor("lidar", LidarConverter)
        >>> # Sekarang bisa digunakan: brain.perceive(lidar_data, modality="lidar")
    """
    from resonait.converters.base import BaseConverter
    if not issubclass(converter_class, BaseConverter):
        raise TypeError(
            f"Sensor '{name}' harus mewarisi dari BaseConverter. "
            f"Lihat dokumentasi di resonait/converters/base.py"
        )
    SENSOR_REGISTRY[name] = converter_class
    print(f"[ResonAIt] ✓ Sensor '{name}' berhasil didaftarkan.")

# Auto-load sensor bawaan
def _load_default_sensors():
    """Muat sensor default saat package pertama kali diimport."""
    from resonait.converters.text_converter import TextConverter
    from resonait.converters.image_converter import ImageConverter
    from resonait.converters.audio_converter import AudioConverter
    
    SENSOR_REGISTRY["text"]  = TextConverter
    SENSOR_REGISTRY["image"] = ImageConverter
    SENSOR_REGISTRY["audio"] = AudioConverter

_load_default_sensors()

__all__ = [
    "ResonAItBrain",
    "FrequencyTensor",
    "UniversalFrequencySpace",
    "UniversalFrequencyConverter",
    "DissonanceEngine",
    "register_sensor",
    "SENSOR_REGISTRY",
    "__version__",
]
