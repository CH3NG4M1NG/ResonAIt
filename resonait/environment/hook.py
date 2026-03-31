"""
resonait/environment/hook.py
=============================
SIMULATED ENVIRONMENT HOOK

API Interface untuk menghubungkan ResonAIt ke mesin game
(Unreal Engine, Unity, Godot, OpenAI Gym, dll).

Arsitektur Integrasi:
    Game Engine ←→ EnvironmentHook ←→ ResonAItBrain
    
    Game engine mengirim:
    - Observasi (gambar frame, posisi, status)
    - Event damage (peluru kena, jatuh, dll)
    - Reward signal
    
    ResonAIt mengirim:
    - Aksi yang dipilih (gerak, tembak, hindari)
    - Reasoning (mengapa memilih aksi ini)
    
Mendukung dua mode koneksi:
    1. Direct API : Import langsung (untuk Python-based simulators)
    2. Network API: REST/WebSocket (untuk Unity/Unreal — mesin non-Python)

Contoh integrasi Unity:
    [Unity C#] → HTTP POST /observe → [ResonAIt] → HTTP GET /action → [Unity C#]
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import threading
from pathlib import Path

from resonait.core.frequency_space import FrequencyTensor, Modality
from resonait.pain.dissonance import DissonanceEngine, DamageEvent, DamageType


# ============================================================
# ENUM: Mode Koneksi
# ============================================================

class ConnectionMode(Enum):
    DIRECT  = "direct"   # Import langsung di Python
    REST    = "rest"     # HTTP REST API (Unity/Unreal)
    SOCKET  = "socket"   # WebSocket untuk low-latency


# ============================================================
# DATACLASS: Observasi dari Environment
# ============================================================

@dataclass
class EnvironmentObservation:
    """
    Representasi lengkap satu "frame" observasi dari environment.
    
    Ini adalah semua informasi yang diterima AI dari dunia simulasinya.
    
    Attributes:
        frame_id     : Nomor frame saat ini
        visual_data  : Data gambar/frame dari kamera game (numpy array)
        audio_data   : Data audio dari environment (numpy array)
        agent_state  : Status agen (posisi, health, ammo, dll)
        nearby_objects: Objek-objek di sekitar agen
        damage_events: Event kerusakan yang terjadi di frame ini
        timestamp    : Waktu observasi (detik)
    """
    frame_id:       int
    visual_data:    Optional[np.ndarray] = None   # (H, W, 3) RGB frame
    audio_data:     Optional[np.ndarray] = None   # (samples,) audio mono
    agent_state:    Dict[str, Any] = field(default_factory=dict)
    nearby_objects: List[Dict]     = field(default_factory=list)
    damage_events:  List[DamageEvent] = field(default_factory=list)
    timestamp:      float = field(default_factory=time.time)
    
    def has_visual(self) -> bool:
        return self.visual_data is not None
    
    def has_audio(self) -> bool:
        return self.audio_data is not None
    
    def has_damage(self) -> bool:
        return len(self.damage_events) > 0


# ============================================================
# DATACLASS: Aksi AI
# ============================================================

@dataclass
class AgentAction:
    """
    Aksi yang dipilih AI untuk dikirim ke environment.
    
    Attributes:
        action_id   : ID aksi diskrit (e.g., 0=forward, 1=back, 2=left, ...)
        action_vector: Aksi kontinu (e.g., [0.5, -0.2, 0.0] untuk 3D movement)
        confidence  : Seberapa yakin AI dengan pilihan ini (0.0-1.0)
        reasoning   : Deskripsi singkat mengapa AI memilih aksi ini
        cognitive_state: State kognitif saat keputusan dibuat
    """
    action_id:      int
    action_vector:  Optional[np.ndarray] = None
    confidence:     float = 0.5
    reasoning:      str   = ""
    cognitive_state: Dict = field(default_factory=dict)


# ============================================================
# KELAS UTAMA: EnvironmentHook
# ============================================================

class EnvironmentHook:
    """
    Interface utama untuk menghubungkan ResonAIt ke environment simulasi.
    
    EnvironmentHook bertindak sebagai "mediator" antara:
    - Dunia simulasi (game, robotika, lab virtual)
    - Otak ResonAIt yang memproses persepsi di domain frekuensi
    
    Features:
    - Multimodal perception: Memproses visual + audio + text state sekaligus
    - Pain integration: Otomatis menerapkan damage ke otak via DissonanceEngine
    - Action selection: Mengubah output frekuensi menjadi aksi discrete/continuous
    - Episode management: Reset antar episode, tracking performa
    - Logging: Catat semua interaksi untuk analisis dan training
    
    Args:
        brain     : Instance ResonAItBrain
        n_actions : Jumlah aksi diskrit yang tersedia
        freq_dim  : Dimensi frekuensi (harus sama dengan brain)
        log_dir   : Direktori untuk menyimpan log interaksi
    """
    
    def __init__(
        self,
        brain,              # ResonAItBrain
        n_actions:   int  = 8,
        freq_dim:    int  = 512,
        log_dir:     Optional[str] = None,
    ):
        self.brain     = brain
        self.n_actions = n_actions
        self.freq_dim  = freq_dim
        
        # === KONVERTER MULTIMODAL ===
        # Diinisialisasi secara lazy untuk menghindari circular imports
        self._converter = None
        
        # === PAIN SYSTEM ===
        self.dissonance_engine = DissonanceEngine(freq_dim=freq_dim)
        
        # === ACTION HEAD ===
        # Layer kecil yang memetakan output frekuensi ke distribusi aksi
        self.action_head = torch.nn.Sequential(
            torch.nn.Linear(freq_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, n_actions),
        )
        
        # === EPISODE TRACKING ===
        self.episode_count    = 0
        self.step_count       = 0
        self.episode_reward   = 0.0
        self.episode_history  = []
        
        # === LOGGING ===
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # === CALLBACK REGISTRY ===
        # Komunitas dapat menambahkan callback untuk event tertentu
        self._callbacks: Dict[str, List[Callable]] = {
            "on_observe":  [],
            "on_damage":   [],
            "on_act":      [],
            "on_reset":    [],
            "on_episode_end": [],
        }
        
        print(f"[EnvironmentHook] ✓ Initialized")
        print(f"[EnvironmentHook]   Actions: {n_actions}")
        print(f"[EnvironmentHook]   Freq dim: {freq_dim}")
    
    # ==========================================================
    # LIFECYCLE METHODS
    # ==========================================================
    
    def reset(self, info: Optional[Dict] = None) -> Dict:
        """
        Reset untuk episode baru.
        
        Dipanggil di awal setiap episode game baru.
        Reset pain state, tracking vars, dan notifikasi brain.
        
        Args:
            info: Informasi tentang episode baru (level, difficulty, dll)
            
        Returns:
            Status setelah reset
        """
        # Catat performa episode yang baru selesai
        if self.episode_count > 0:
            episode_summary = {
                "episode":    self.episode_count,
                "steps":      self.step_count,
                "reward":     self.episode_reward,
                "health":     self.brain.get_health_report(),
            }
            self.episode_history.append(episode_summary)
            
            # Panggil on_episode_end callbacks
            for cb in self._callbacks["on_episode_end"]:
                cb(episode_summary)
        
        # Reset semua state
        self.dissonance_engine.reset()
        self.dissonance_engine.step_recovery(self.brain)
        self.brain.pain_state.zero_()
        self.brain.spectral_health.fill_(1.0)
        
        self.episode_count  += 1
        self.step_count      = 0
        self.episode_reward  = 0.0
        
        # Panggil on_reset callbacks
        for cb in self._callbacks["on_reset"]:
            cb({"episode": self.episode_count, "info": info})
        
        print(f"[EnvironmentHook] 🔄 Episode {self.episode_count} dimulai.")
        
        return {"episode": self.episode_count, "pain": 0.0, "health": 1.0}
    
    # ==========================================================
    # CORE LOOP: OBSERVE → THINK → ACT
    # ==========================================================
    
    def observe_and_act(
        self,
        observation: EnvironmentObservation,
        reward:      float = 0.0,
    ) -> AgentAction:
        """
        FUNGSI UTAMA LOOP: Observasi → Persepsi → Pemikiran → Aksi
        
        Ini adalah fungsi yang dipanggil setiap step simulasi.
        
        Alur lengkap:
        1. Konversi observasi multimodal → FrequencyTensors
        2. Proses damage events → injeksi pain ke otak
        3. Fusi semua FrequencyTensors menjadi satu persepsi
        4. Otak memproses persepsi frekuensi
        5. Pilih aksi dari output otak
        6. Recovery parsial dari pain
        7. Log dan return aksi
        
        Args:
            observation: Observasi lengkap dari environment
            reward     : Reward signal dari environment (untuk tracking)
            
        Returns:
            AgentAction yang harus dieksekusi di environment
        """
        self.step_count    += 1
        self.episode_reward += reward
        
        # === Step 1: Inisialisasi converter ===
        if self._converter is None:
            from resonait.converters.universal_converter import UniversalFrequencyConverter
            self._converter = UniversalFrequencyConverter(freq_dim=self.freq_dim)
        
        # === Step 2: Konversi Observasi Multimodal → FrequencyTensors ===
        freq_tensors = self._multimodal_perception(observation)
        
        # === Step 3: Proses Damage Events ===
        damage_reports = []
        if observation.has_damage():
            for damage_event in observation.damage_events:
                report = self.dissonance_engine.apply_to_brain(
                    self.brain, damage_event
                )
                damage_reports.append(report)
                
                # Panggil on_damage callbacks
                for cb in self._callbacks["on_damage"]:
                    cb(damage_event, report)
        
        # === Step 4: Fusi Multimodal Perception ===
        fused_freq = self._fuse_multimodal(freq_tensors)
        
        # === Step 5: Otak memproses persepsi ===
        with torch.no_grad():  # Inference, bukan training
            brain_output = self.brain(fused_freq)
        
        # === Step 6: Pilih Aksi ===
        action = self._select_action(brain_output, observation)
        
        # === Step 7: Recovery parsial ===
        # Setiap step tanpa damage = sedikit recovery
        if not observation.has_damage():
            self.dissonance_engine.step_recovery(self.brain)
        
        # === Step 8: Logging ===
        if self.log_dir:
            self._log_step(observation, action, brain_output, damage_reports)
        
        # Panggil on_act callbacks
        for cb in self._callbacks["on_act"]:
            cb(observation, action)
        
        return action
    
    def _multimodal_perception(
        self,
        obs: EnvironmentObservation
    ) -> Dict[str, FrequencyTensor]:
        """
        Konversi semua modalitas dalam observasi ke FrequencyTensors.
        
        Memproses visual, audio, dan state teks secara paralel.
        """
        freq_tensors = {}
        
        # Visual perception
        if obs.has_visual():
            try:
                freq_tensors["visual"] = self._converter.convert(
                    obs.visual_data, modality="image"
                )
            except Exception as e:
                print(f"[EnvironmentHook] ⚠ Gagal memproses visual: {e}")
        
        # Audio perception
        if obs.has_audio():
            try:
                freq_tensors["audio"] = self._converter.convert(
                    obs.audio_data, modality="audio"
                )
            except Exception as e:
                print(f"[EnvironmentHook] ⚠ Gagal memproses audio: {e}")
        
        # State perception (teks/numerik)
        if obs.agent_state:
            state_text = json.dumps(obs.agent_state)
            freq_tensors["state"] = self._converter.convert(
                state_text, modality="text"
            )
        
        return freq_tensors
    
    def _fuse_multimodal(
        self,
        freq_tensors: Dict[str, FrequencyTensor]
    ) -> FrequencyTensor:
        """
        Fusi beberapa FrequencyTensor menjadi satu representasi terpadu.
        
        Fusi dilakukan via superposisi gelombang — sama seperti
        cara otak manusia mengintegrasikan berbagai indera.
        
        Jika hanya ada satu modalitas, langsung return tensor tersebut.
        Jika banyak modalitas, interferensikan satu per satu.
        """
        if not freq_tensors:
            # Fallback: buat tensor kosong
            zeros = torch.zeros(1, 1, self.freq_dim)
            from resonait.core.frequency_space import FrequencyTensor, Modality
            return FrequencyTensor(amplitude=zeros, phase=zeros, modality=Modality.CUSTOM)
        
        tensors_list = list(freq_tensors.values())
        
        if len(tensors_list) == 1:
            return tensors_list[0]
        
        # Superposisi semua modalitas
        fused = tensors_list[0]
        for tensor in tensors_list[1:]:
            # Pastikan shape sama (ambil minimal)
            if fused.amplitude.shape == tensor.amplitude.shape:
                fused = fused.interfere_with(tensor)
        
        return fused
    
    def _select_action(
        self,
        brain_output: Dict,
        obs: EnvironmentObservation,
    ) -> AgentAction:
        """
        Konversi output otak menjadi aksi yang executable.
        
        Output otak adalah tensor di domain frekuensi.
        Kita proyeksikan ke distribusi probabilitas aksi.
        """
        output_tensor = brain_output["output"]  # (batch, channels, length)
        
        # Pool over channel dan length → vektor ringkasan
        pooled = output_tensor.mean(dim=[0, 2])  # (hidden_dim,)
        
        # Resize ke freq_dim jika perlu
        if pooled.shape[0] != self.freq_dim:
            pooled = torch.nn.functional.interpolate(
                pooled.unsqueeze(0).unsqueeze(0),
                size=self.freq_dim,
                mode='linear',
                align_corners=False,
            ).squeeze()
        
        # Proyeksi ke logit aksi
        action_logits = self.action_head(pooled.detach())  # (n_actions,)
        
        # Sampling dari distribusi softmax
        action_probs    = torch.softmax(action_logits, dim=-1)
        action_id       = torch.multinomial(action_probs, num_samples=1).item()
        confidence      = action_probs[action_id].item()
        
        # Buat action vector kontinu (berguna untuk game dengan aksi kontinu)
        action_vector = action_probs.detach().cpu().numpy()
        
        # Buat reasoning ringkas
        pain_level   = brain_output.get("pain_level", 0.0)
        logic_gate   = brain_output.get("logic_gate", 0.0)
        
        reasoning = (
            f"Action {action_id} (conf={confidence:.2f}), "
            f"logic={logic_gate:.2f}, pain={pain_level:.2f}"
        )
        
        return AgentAction(
            action_id=action_id,
            action_vector=action_vector,
            confidence=confidence,
            reasoning=reasoning,
            cognitive_state={
                "logic_gate":       brain_output.get("logic_gate", 0),
                "imagination_gate": brain_output.get("imagination_gate", 0),
                "memory_gate":      brain_output.get("memory_gate", 0),
                "pain_level":       pain_level,
            }
        )
    
    def _log_step(
        self,
        obs:      EnvironmentObservation,
        action:   AgentAction,
        output:   Dict,
        damages:  List,
    ):
        """Log satu step ke file untuk analisis."""
        log_entry = {
            "episode":   self.episode_count,
            "step":      self.step_count,
            "reward":    self.episode_reward,
            "action_id": action.action_id,
            "confidence": action.confidence,
            "pain_level": output.get("pain_level", 0.0),
            "damages":   len(damages),
            "health":    self.brain.get_health_report(),
        }
        
        log_path = self.log_dir / f"episode_{self.episode_count}.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    # ==========================================================
    # CALLBACK SYSTEM (Extension Point Komunitas)
    # ==========================================================
    
    def register_callback(self, event: str, callback: Callable):
        """
        Daftarkan callback untuk event tertentu.
        
        Extension point untuk komunitas — bisa menambahkan custom logic
        tanpa memodifikasi kode inti.
        
        Events yang tersedia:
        - "on_observe"    : Dipanggil setiap observasi baru
        - "on_damage"     : Dipanggil saat AI menerima damage
        - "on_act"        : Dipanggil saat AI memilih aksi
        - "on_reset"      : Dipanggil saat episode baru dimulai
        - "on_episode_end": Dipanggil di akhir episode
        
        Contoh:
            def log_damage(event, report):
                print(f"DAMAGE! Type: {event.damage_type}, Pain: {report['accumulated_pain']:.2f}")
            
            hook.register_callback("on_damage", log_damage)
        """
        if event not in self._callbacks:
            raise ValueError(f"Event '{event}' tidak dikenal. Tersedia: {list(self._callbacks.keys())}")
        
        self._callbacks[event].append(callback)
        print(f"[EnvironmentHook] ✓ Callback terdaftar untuk event '{event}'.")
    
    # ==========================================================
    # NETWORK API (untuk Unity/Unreal — non-Python engines)
    # ==========================================================
    
    def start_network_server(self, host: str = "localhost", port: int = 8765):
        """
        Mulai server HTTP sederhana untuk terima observasi dari Unity/Unreal.
        
        Endpoint:
        - POST /observe  : Kirim observasi, terima aksi
        - GET  /health   : Cek status otak
        - POST /damage   : Kirim damage event
        - POST /reset    : Reset episode
        
        Integrasi Unity:
            Dari C#, kirim HTTP POST ke http://localhost:8765/observe
            dengan JSON berisi observasi, dan terima JSON aksi sebagai response.
        
        Args:
            host: Host untuk server (default: localhost)
            port: Port untuk server (default: 8765)
        """
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            raise ImportError(
                "Flask diperlukan untuk network server: pip install flask\n"
                "Atau gunakan direct API jika environment berbasis Python."
            )
        
        app = Flask("ResonAIt-EnvironmentHook")
        hook_self = self  # Reference ke self untuk closure
        
        @app.route("/observe", methods=["POST"])
        def observe():
            """Endpoint observasi utama dari game engine."""
            data = request.json
            
            # Parse observasi dari JSON
            obs = EnvironmentObservation(
                frame_id=data.get("frame_id", 0),
                agent_state=data.get("agent_state", {}),
                damage_events=[
                    DamageEvent(
                        damage_type=DamageType[d["type"].upper()],
                        intensity=d["intensity"],
                    )
                    for d in data.get("damages", [])
                ]
            )
            
            # Proses dan dapatkan aksi
            action = hook_self.observe_and_act(obs, reward=data.get("reward", 0.0))
            
            return jsonify({
                "action_id":    action.action_id,
                "confidence":   action.confidence,
                "reasoning":    action.reasoning,
                "cognitive_state": action.cognitive_state,
            })
        
        @app.route("/health", methods=["GET"])
        def health():
            """Cek status kesehatan otak AI."""
            return jsonify(hook_self.brain.get_health_report())
        
        @app.route("/reset", methods=["POST"])
        def reset():
            """Reset untuk episode baru."""
            result = hook_self.reset(request.json)
            return jsonify(result)
        
        print(f"[EnvironmentHook] 🌐 Network server dimulai di http://{host}:{port}")
        print(f"[EnvironmentHook]   POST /observe  → Kirim observasi, terima aksi")
        print(f"[EnvironmentHook]   GET  /health   → Status otak")
        print(f"[EnvironmentHook]   POST /reset    → Reset episode")
        
        # Jalankan di thread terpisah agar tidak memblokir
        server_thread = threading.Thread(
            target=lambda: app.run(host=host, port=port, debug=False),
            daemon=True
        )
        server_thread.start()
        
        return app


def main_cli():
    """CLI untuk menjalankan environment hook sebagai server standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ResonAIt Environment Hook Server")
    parser.add_argument("--brain", type=str, required=True,
                        help="Path ke checkpoint ResonAItBrain")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--actions", type=int, default=8)
    
    args = parser.parse_args()
    
    from resonait.core.brain import ResonAItBrain
    brain = ResonAItBrain.load(args.brain)
    hook  = EnvironmentHook(brain=brain, n_actions=args.actions)
    hook.start_network_server(host=args.host, port=args.port)
    
    print(f"\n[CLI] Server berjalan. Tekan Ctrl+C untuk berhenti.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[CLI] Server dihentikan.")


if __name__ == "__main__":
    main_cli()
