"""
examples/train_unified.py
===========================
CONTOH LENGKAP: Dari model-model terpisah → Satu Unified Model

Script ini menunjukkan cara menggunakan ResonAIt untuk:
1. Mengambil model specialist (Qwen, Whisper, dll)
2. Meng-align semuanya ke Universal Frequency Space
3. Menghasilkan SATU model yang bisa melakukan semua task

===============================================================
VISI:
    Kamu ambil Qwen 2.5 3B (chatbot) +
           Stable Diffusion (image gen) +
           Whisper small (speech-to-text) +
           MusicGen small (music gen)
    
    → Setelah training: SATU model bisa chat, generate gambar,
      transkrip audio, dan buat musik sekaligus
    → Semua modalitas "bicara" dalam satu bahasa: Frekuensi
===============================================================

Jalankan dengan:
    python examples/train_unified.py

Untuk Kaggle T4 x2 (16GB x2):
    python examples/train_unified.py --preset kaggle

Untuk testing tanpa GPU:
    python examples/train_unified.py --preset cpu_debug
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def get_preset_config(preset: str):
    """
    Konfigurasi siap pakai untuk berbagai setup hardware.
    """
    from resonait.tools.unified_trainer import UnifiedTrainerConfig

    presets = {

        # ── Untuk Kaggle T4 x2 (VRAM 2×16GB) ─────────────────────
        "kaggle": UnifiedTrainerConfig(
            specialist_models={
                "llm": "Qwen/Qwen2.5-3B",        # Chatbot specialist
                "asr": "openai/whisper-small",    # ASR specialist (bisa juga dijadikan TTS)
            },
            freq_dim     = 512,
            hidden_dim   = 256,
            n_modes      = 64,
            n_fno_layers = 4,
            image_size   = 128,   # Lebih kecil untuk hemat VRAM
            n_frames     = 8,     # Lebih sedikit frame video
            phase1_steps = 500,
            phase2_steps = 1000,
            phase3_steps = 300,
            phase4_steps = 300,
            phase5_steps = 500,
            batch_size   = 4,
            learning_rate= 2e-4,
            precision    = "fp16",
            device       = "auto",
            output_dir   = "./resonait_unified_kaggle",
        ),

        # ── Untuk GPU besar (A100/H100) ────────────────────────────
        "full": UnifiedTrainerConfig(
            specialist_models={
                "llm":      "Qwen/Qwen2.5-7B",
                "image_gen":"stabilityai/stable-diffusion-3.5-medium",
                "asr":      "openai/whisper-large-v3",
                "music":    "facebook/musicgen-small",
            },
            freq_dim     = 1024,
            hidden_dim   = 512,
            n_modes      = 128,
            n_fno_layers = 6,
            image_size   = 256,
            n_frames     = 16,
            phase1_steps = 2000,
            phase2_steps = 5000,
            phase3_steps = 1000,
            phase4_steps = 1000,
            phase5_steps = 2000,
            batch_size   = 8,
            learning_rate= 1e-4,
            precision    = "bf16",
            device       = "auto",
            output_dir   = "./resonait_unified_full",
        ),

        # ── Untuk CPU / debug (tanpa GPU, no internet) ─────────────
        "cpu_debug": UnifiedTrainerConfig(
            specialist_models={},   # Kosong — tidak perlu download model
            freq_dim     = 128,
            hidden_dim   = 64,
            n_modes      = 16,
            n_fno_layers = 1,
            image_size   = 64,
            n_frames     = 4,
            phase1_steps = 0,       # Skip phase1 (tidak ada specialist)
            phase2_steps = 20,
            phase3_steps = 10,
            phase4_steps = 10,
            phase5_steps = 10,
            batch_size   = 2,
            learning_rate= 1e-3,
            precision    = "fp32",
            device       = "cpu",
            output_dir   = "./resonait_unified_debug",
            run_phase1_alignment = False,
        ),
    }

    return presets.get(preset, presets["cpu_debug"])


def demo_inference(model_path: str):
    """
    Demo: gunakan unified model yang sudah dilatih untuk berbagai task.
    """
    from resonait.core.unified_model import ResonAItUnified, UnifiedInput, TaskType

    print(f"\n{'='*60}")
    print(f"  DEMO INFERENCE — Unified Model")
    print(f"{'='*60}")

    # Load model
    model = ResonAItUnified.from_checkpoint(model_path)
    model.eval()

    print(model.get_model_card())

    # ── Demo 1: Chat ────────────────────────────────────────────
    print("\n[Demo 1] Chat / Text Generation")
    result = model.run(UnifiedInput(
        task=TaskType.CHAT,
        text_prompt="Jelaskan cara kerja Fourier Transform dalam 3 kalimat.",
    ))
    print(f"  Prompt : 'Jelaskan cara kerja Fourier Transform...'")
    print(f"  Output : {result.text_output}")
    print(f"  Decoder: {result.metadata.get('decoder_used', '?')}")

    # ── Demo 2: Image Generation ────────────────────────────────
    print("\n[Demo 2] Text-to-Image")
    result = model.run(UnifiedInput(
        task=TaskType.TEXT_TO_IMAGE,
        text_prompt="Pemandangan gunung berapi di Jawa dengan latar matahari terbenam",
        output_config={"size": 256}
    ))
    if result.image_output is not None:
        print(f"  Output : numpy array {result.image_output.shape} (H,W,C)")
        print(f"  Info   : {result.metadata}")

    # ── Demo 3: TTS ─────────────────────────────────────────────
    print("\n[Demo 3] Text-to-Speech")
    result = model.run(UnifiedInput(
        task=TaskType.TEXT_TO_SPEECH,
        text_prompt="Selamat datang di ResonAIt, model AGI masa depan.",
    ))
    if result.audio_output is not None:
        sr       = result.metadata.get("sample_rate", 22050)
        duration = result.metadata.get("duration_s", 0)
        print(f"  Output : waveform numpy {result.audio_output.shape}")
        print(f"  Sample rate: {sr} Hz | Durasi: {duration:.1f}s")

    # ── Demo 4: Video Generation ────────────────────────────────
    print("\n[Demo 4] Text-to-Video")
    result = model.run(UnifiedInput(
        task=TaskType.TEXT_TO_VIDEO,
        text_prompt="Seekor kucing berlari di pantai saat matahari terbenam",
        output_config={"fps": 8, "duration": 2}
    ))
    if result.video_output is not None:
        n_frames = len(result.video_output)
        print(f"  Output : {n_frames} frames @ {result.metadata.get('fps',8)} fps")
        print(f"  Durasi : {result.metadata.get('duration_s', 0):.1f}s")

    # ── Demo 5: Auto task detection ─────────────────────────────
    print("\n[Demo 5] Auto Task Detection")
    result = model.run(UnifiedInput(
        task=TaskType.AUTO,
        text_prompt="Buatkan gambar bunga matahari di ladang",
    ))
    print(f"  Task terdeteksi: {result.metadata.get('task_detected', '?')}")
    print(f"  Decoder dipakai: {result.metadata.get('decoder_used', '?')}")

    print(f"\n{'='*60}")
    print(f"  Demo selesai! Model berjalan dengan {len(ResonAItUnified.TASK_LIST)} task.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ResonAIt Unified Training Pipeline"
    )
    parser.add_argument(
        "--preset", type=str, default="cpu_debug",
        choices=["kaggle", "full", "cpu_debug"],
        help="Pilihan preset hardware (default: cpu_debug)"
    )
    parser.add_argument(
        "--demo-only", type=str, default=None,
        metavar="CHECKPOINT_PATH",
        help="Skip training, langsung demo inference dari checkpoint"
    )
    parser.add_argument(
        "--skip-phase", type=int, nargs="+", default=[],
        help="Skip phase tertentu (1-5)"
    )
    args = parser.parse_args()

    # ── Mode: Demo saja ─────────────────────────────────────────
    if args.demo_only:
        demo_inference(args.demo_only)
        return

    # ── Mode: Full training ─────────────────────────────────────
    from resonait.tools.unified_trainer import UnifiedTrainer

    config = get_preset_config(args.preset)

    # Apply skip phases
    if 1 in args.skip_phase:
        config.run_phase1_alignment   = False
    if 2 in args.skip_phase:
        config.run_phase2_pretraining = False
    if 3 in args.skip_phase:
        config.run_phase3_contrastive = False
    if 4 in args.skip_phase:
        config.run_phase4_task_ft     = False
    if 5 in args.skip_phase:
        config.run_phase5_joint       = False

    print(f"\n  Preset   : {args.preset}")
    print(f"  Specialists: {list(config.specialist_models.keys()) or ['None (debug mode)']}")
    print(f"  Device   : {config.device}")
    print(f"  Output   : {config.output_dir}")

    # Jalankan training
    trainer = UnifiedTrainer(config)
    trainer.run()

    # Demo setelah training
    final_model_path = f"{config.output_dir}/final_unified_model.pt"
    if os.path.exists(final_model_path):
        demo_inference(final_model_path)


if __name__ == "__main__":
    main()
