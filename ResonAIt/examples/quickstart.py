"""
examples/quickstart.py
========================
Contoh penggunaan cepat ResonAIt — dari instalasi hingga inference.

Jalankan dengan: python examples/quickstart.py
"""

import torch
import numpy as np
import sys
sys.path.insert(0, "..")  # Pastikan resonait bisa diimport

print("=" * 60)
print("  RESONAIT — QUICKSTART DEMO")
print("=" * 60)


# ============================================================
# DEMO 1: Universal Frequency Converter
# ============================================================
print("\n[1/4] Universal Frequency Converter")
print("-" * 40)

from resonait.converters.universal_converter import UniversalFrequencyConverter
from resonait.converters.text_converter import TextConverter
from resonait.converters.image_converter import ImageConverter, AudioConverter

# Inisialisasi converter
converter = UniversalFrequencyConverter(freq_dim=512)

# Konversi teks
text_freq = converter.convert("Saya sedang belajar tentang frekuensi AI!")
print(f"✓ Teks → FrequencyTensor: {text_freq}")
print(f"  Amplitudo max: {text_freq.amplitude.max().item():.4f}")
print(f"  Frekuensi dominan (top-5): {text_freq.dominant_frequencies(5)[0, 0, :]}")

# Konversi gambar sintetis
fake_image = np.random.rand(64, 64, 3).astype(np.float32)
image_freq = converter.convert(fake_image, modality="image")
print(f"✓ Gambar (64x64 RGB) → FrequencyTensor: {image_freq}")

# Konversi audio sintetis (1 detik @ 22050 Hz)
fake_audio = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))).astype(np.float32)
audio_freq = converter.convert(fake_audio, modality="audio")
print(f"✓ Audio (1 detik, 440 Hz) → FrequencyTensor: {audio_freq}")

# Hitung koherensi antara teks dan gambar
coherence = text_freq.coherence_with(image_freq)
print(f"\n  Koherensi Teks↔Gambar: {coherence:.4f} (nilai rendah = berbeda modalitas)")


# ============================================================
# DEMO 2: ResonAIt Brain — Proses Multimodal
# ============================================================
print("\n[2/4] ResonAIt Brain — Fourier Neural Operator")
print("-" * 40)

from resonait.core.brain import ResonAItBrain

# Buat otak dengan konfigurasi kecil untuk demo
brain = ResonAItBrain(
    freq_dim=512,
    hidden_dim=64,    # Kecil untuk demo
    n_modes=32,
    n_fno_layers=2,
)

print(f"✓ ResonAItBrain dibuat!")
print(f"  Parameter: {sum(p.numel() for p in brain.parameters()):,}")

# Proses teks
output = brain(text_freq)
print(f"\n  Output dari teks:")
print(f"  - Logic gate   : {output['logic_gate']:.4f}")
print(f"  - Imagination  : {output['imagination_gate']:.4f}")
print(f"  - Memory gate  : {output['memory_gate']:.4f}")
print(f"  - Pain level   : {output['pain_level']:.4f}")

# Proses gambar
output_img = brain(image_freq)
print(f"\n  Output dari gambar:")
print(f"  - Logic gate   : {output_img['logic_gate']:.4f}")
print(f"  - Imagination  : {output_img['imagination_gate']:.4f}")


# ============================================================
# DEMO 3: Pain System — Interferensi Destruktif
# ============================================================
print("\n[3/4] Pain System — Frequency-Based Dissonance")
print("-" * 40)

from resonait.pain.dissonance import DissonanceEngine, DamageEvent, DamageType

engine = DissonanceEngine(freq_dim=512)

# Kondisi awal
health_before = brain.get_health_report()
print(f"  Kesehatan SEBELUM damage: {health_before}")

# Simulasi damage dari game (seperti kena tembak di PUBG)
damage = DamageEvent(
    damage_type=DamageType.PHYSICAL,
    intensity=0.7,          # 70% damage
    duration_ms=200,
    source_position=(10.0, 0.0, 5.0),
    metadata={"weapon": "M416", "damage_points": 70}
)

report = engine.apply_to_brain(brain, damage)
print(f"\n  💥 DAMAGE DITERIMA! Type: PHYSICAL, Intensity: 0.7")
print(f"  Damage report: {report}")

health_after = brain.get_health_report()
print(f"\n  Kesehatan SESUDAH damage: {health_after}")

# Proses teks LAGI — perhatikan perubahan output karena pain
output_pain = brain(text_freq)
print(f"\n  Output saat dalam pain:")
print(f"  - Pain level   : {output_pain['pain_level']:.4f} (vs {output['pain_level']:.4f} sebelum)")
print(f"  - Logic gate   : {output_pain['logic_gate']:.4f}")

# Recovery
print(f"\n  ⏳ Recovery...")
for i in range(10):
    engine.step_recovery(brain)

health_recovered = brain.get_health_report()
print(f"  Kesehatan SETELAH recovery: {health_recovered}")


# ============================================================
# DEMO 4: Autonomous Data Sourcing
# ============================================================
print("\n[4/4] Autonomous Data Sourcing — Ekspansi Multimodal")
print("-" * 40)

texts = [
    "Kucing sedang berlari di taman.",
    "Mobil melintas dengan kecepatan tinggi.",
    "Hujan turun dengan deras di malam hari.",
]

print(f"  Input: {len(texts)} teks")
print(f"  Strategy: simulate (offline)")

expanded = converter.autonomous_multimodal_expansion(texts, strategy="simulate")

for i, item in enumerate(expanded):
    print(f"\n  Teks {i+1}: '{texts[i][:30]}...'")
    print(f"    ✓ text  freq: {item['text'].shape}")
    print(f"    ✓ image freq: {item['image'].shape}")
    print(f"    ✓ audio freq: {item['audio'].shape}")

print(f"\n  → {len(expanded)} sampel multimodal siap untuk alignment training!")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  DEMO SELESAI!")
print("=" * 60)
print("""
  ResonAIt siap digunakan. Langkah selanjutnya:
  
  1. ALIGNMENT: Upgrade LLM privat ke frequency space
     >>> from resonait.tools.alignment import LLMAlignmentTool, AlignmentConfig
     >>> config = AlignmentConfig(llm_model_name="mistralai/Mistral-7B-v0.1")
     >>> tool = LLMAlignmentTool(config)
     >>> tool.load_source_llm()
     >>> tool.align(training_texts)
  
  2. GAME INTEGRATION: Hubungkan ke Unity/Unreal
     >>> from resonait.environment.hook import EnvironmentHook
     >>> hook = EnvironmentHook(brain=brain, n_actions=8)
     >>> hook.start_network_server(port=8765)
     # → Dari Unity C#, POST ke http://localhost:8765/observe
  
  3. SENSOR BARU: Tambahkan modalitas custom
     >>> from resonait import register_sensor
     >>> from my_plugin import LidarConverter
     >>> register_sensor("lidar", LidarConverter)
     >>> freq = converter.convert(lidar_data, modality="lidar")
  
  Dokumentasi lengkap: https://resonait.readthedocs.io
""")
