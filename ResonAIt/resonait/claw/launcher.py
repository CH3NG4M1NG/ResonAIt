"""
resonait/claw/launcher.py
==========================
RESONAITCLAW LAUNCHER — Jalankan Claw di terminal

Usage:
    python -m resonait.claw.launcher --checkpoint ./claw_checkpoints
    python -m resonait.claw.launcher --new
    python -m resonait.claw.launcher --help

Claw akan berjalan interaktif di terminal, aktif di background,
dan bisa menerima input teks, path file, atau perintah khusus.
"""

import argparse
import sys
import time
import threading
import os
from pathlib import Path


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║          R E S O N A I T  C L A W                   ║
║      Active AGI — Universal Frequency Space          ║
╠══════════════════════════════════════════════════════╣
║  Commands:                                           ║
║    /status   — lihat state Claw saat ini             ║
║    /mood     — mood dan emosi Claw                   ║
║    /memory   — ringkasan memori                      ║
║    /help     — tampilkan bantuan                     ║
║    /save     — simpan state                          ║
║    /quit     — keluar                                ║
╚══════════════════════════════════════════════════════╝
""")


def format_response(response, show_meta: bool = False) -> str:
    """Format respons Claw untuk ditampilkan di terminal."""
    lines = [f"\n  Claw [{response.emotion}]: {response.content}"]
    
    if show_meta and response.metadata:
        meta = response.metadata
        lines.append(
            f"  [logic={meta.get('logic_gate',0):.2f} "
            f"imag={meta.get('imagination',0):.2f} "
            f"mem={meta.get('memory_gate',0):.2f}]"
        )
    
    return "\n".join(lines)


def handle_command(cmd: str, claw):
    """Tangani perintah khusus (dimulai dengan /)."""
    from resonait.claw.emotion_engine import EmotionType
    
    cmd = cmd.strip().lower()
    
    if cmd == '/status':
        status = claw.get_status()
        lines = ["\n  === STATUS ==="]
        for k, v in status.items():
            if not isinstance(v, dict):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    
    elif cmd == '/mood':
        if claw.emotion_engine:
            desc = claw.emotion_engine.get_mood_description()
            state = claw.emotion_engine.current_state.as_text()
            return f"\n  Mood: {claw.mood}\n  State: {state}\n  {desc}"
        return "\n  Emotion engine tidak aktif."
    
    elif cmd == '/memory':
        if claw.memory:
            stats = claw.memory.get_stats()
            return (
                f"\n  Memory Stats:\n"
                f"  STM: {stats['stm']['size']}/{stats['stm']['capacity']}\n"
                f"  LTM: {stats['ltm']['stored_memories']} items\n"
                f"  Utilization: {stats['ltm']['utilization']}"
            )
        return "\n  Memory system tidak aktif."
    
    elif cmd == '/save':
        claw.save()
        return "\n  ✅ State disimpan."
    
    elif cmd == '/help':
        return """
  Commands:
    /status  — state Claw (mood, memory, health)
    /mood    — detail emosi dan arousal
    /memory  — statistik memory STM & LTM
    /save    — simpan ke checkpoint
    /quit    — keluar dari Claw
    
  Tips:
    - Ketik pesan biasa untuk chat
    - Awali dengan "[image]" untuk mode gambar
    - Awali dengan "[music]" untuk generate musik
    - Awali dengan "[code]" untuk mode coding
"""
    
    return None  # Bukan command yang dikenal


def run_interactive(claw, show_meta: bool = False):
    """
    Loop interaktif utama di terminal.
    
    Claw menjawab input user dan juga bisa mengirim inisiatif
    secara proaktif dari background thread.
    """
    # Daftarkan initiative callback
    initiative_shown = threading.Event()
    
    def show_initiative(initiative):
        # Print inisiatif tanpa mengganggu input yang sedang diketik
        print(f"\n\n  Claw [initiative/{initiative.get('emotion','?')}]: {initiative['content']}")
        print("  > ", end='', flush=True)
    
    claw.on_initiative(show_initiative)
    
    # Start background loop
    claw.start_background_loop(interval_s=15.0)
    
    print(f"\n  Ketik sesuatu untuk mulai berbicara dengan {claw.name}.")
    print(f"  Mood awal: {claw.mood}\n")
    
    try:
        while True:
            try:
                user_input = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n\n  Sampai jumpa! Menyimpan state...")
                claw.save()
                break
            
            if not user_input:
                continue
            
            # Perintah khusus
            if user_input.startswith('/'):
                if user_input.lower() == '/quit':
                    print(f"\n  Sampai jumpa! Menyimpan state...")
                    claw.save()
                    break
                
                result = handle_command(user_input, claw)
                if result:
                    print(result)
                else:
                    print(f"  Perintah tidak dikenal: {user_input}")
                continue
            
            # Deteksi prefix task
            from resonait.core.unified_model import TaskType
            task = TaskType.AUTO
            
            if user_input.lower().startswith('[image]'):
                task = TaskType.TEXT_TO_IMAGE
                user_input = user_input[7:].strip()
            elif user_input.lower().startswith('[music]'):
                task = TaskType.TEXT_TO_MUSIC
                user_input = user_input[7:].strip()
            elif user_input.lower().startswith('[code]'):
                task = TaskType.CODE
                user_input = user_input[6:].strip()
            elif user_input.lower().startswith('[speak]'):
                task = TaskType.TEXT_TO_SPEECH
                user_input = user_input[7:].strip()
            
            # Chat
            print(f"  [thinking...]", end='\r')
            t0 = time.time()
            
            response = claw.chat(user_input, task=task)
            
            elapsed = time.time() - t0
            print(format_response(response, show_meta=show_meta))
            print(f"  [{elapsed*1000:.0f}ms | mood: {claw.mood}]")
    
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        claw.stop_background_loop()


def main():
    parser = argparse.ArgumentParser(
        description="ResonAItClaw — Active AGI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mulai dari awal (setup baru)
  python -m resonait.claw.launcher --new
  
  # Load dari checkpoint yang ada
  python -m resonait.claw.launcher --checkpoint ./claw_checkpoints
  
  # Dengan LLM specialist (perlu internet + ~6GB)
  python -m resonait.claw.launcher --checkpoint ./claw_checkpoints --with-llm
  
  # Mode debug (tampilkan metadata tiap respons)
  python -m resonait.claw.launcher --checkpoint ./claw_checkpoints --meta
        """
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default='./claw_checkpoints',
        help='Path ke direktori checkpoint (default: ./claw_checkpoints)'
    )
    parser.add_argument(
        '--new', '-n',
        action='store_true',
        help='Mulai dari awal tanpa checkpoint'
    )
    parser.add_argument(
        '--with-llm',
        action='store_true',
        help='Load specialist LLM (Qwen2.5-3B, butuh ~6GB VRAM)'
    )
    parser.add_argument(
        '--meta',
        action='store_true',
        help='Tampilkan metadata (gate values) tiap respons'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='Claw',
        help='Nama Claw (default: Claw)'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default='~',
        help='Base path untuk environment observer (default: ~)'
    )
    
    args = parser.parse_args()
    
    # Import di sini agar tidak slow jika hanya minta --help
    from resonait.claw.claw import ResonAItClaw, ClawConfig
    
    print_banner()
    
    try:
        if args.new or not Path(args.checkpoint).exists():
            print(f"  Membuat Claw baru...")
            config = ClawConfig(
                name       = args.name,
                checkpoint_dir = args.checkpoint,
                base_path  = args.base_path,
            )
            claw = ResonAItClaw(config)
            claw.setup(load_specialists=args.with_llm)
        else:
            print(f"  Memuat Claw dari: {args.checkpoint}")
            claw = ResonAItClaw.from_checkpoint(
                args.checkpoint,
                load_specialists=args.with_llm,
            )
            claw.config.name = args.name
        
        run_interactive(claw, show_meta=args.meta)
    
    except KeyboardInterrupt:
        print("\n\n  Dihentikan oleh user.")
    except Exception as e:
        print(f"\n  Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
