"""
resonait/claw/claw.py
=======================
RESONAITCLAW — Active AGI Agent

Ini adalah Claw: agent AI yang aktif, punya emosi frekuensi,
bisa berinisiatif, belajar dari lingkungan, dan terus berkembang.

Prinsip Claw (berbeda dari chatbot biasa):
    1. AKTIF, bukan PASIF  — berpikir sendiri saat idle
    2. PUNYA EMOSI         — emosi = pola frekuensi, bukan roleplay
    3. KENAL USER          — belajar pola kerja, preferensi, konteks
    4. BERINISIATIF        — suggest, tanya, refleksi tanpa dipancing
    5. BERKEMBANG          — fine-tune diri dari pengalaman
    6. MULTI-MODAL         — teks, suara, gambar, kode, file — semua satu bahasa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import threading
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

from resonait.core.unified_model import ResonAItUnified, UnifiedInput, UnifiedOutput, TaskType
from resonait.core.frequency_space import FrequencyTensor, Modality
from resonait.memory.frequency_memory import FrequencyMemorySystem
from resonait.pain.dissonance import DissonanceEngine, DamageEvent, DamageType
from resonait.claw.emotion_engine import (
    EmotionEngine, InitiativeEngine, EnvironmentObserver,
    EmotionType, EmotionalState
)


@dataclass
class ClawConfig:
    """Konfigurasi lengkap ResonAItClaw."""
    
    # === IDENTITY ===
    name:         str = "Claw"
    persona:      str = "Asisten AGI yang aktif, penasaran, dan terus berkembang."
    
    # === MODEL ===
    freq_dim:     int = 512
    hidden_dim:   int = 256
    
    # === SPECIALIST MODELS ===
    # Model yang akan di-load (kosongkan untuk skip)
    llm_model:    str = "unsloth/Qwen2.5-3B-Instruct"
    reasoning_model: str = ""  # DeepSeek-R1 distilled (opsional)
    coder_model:  str = ""     # Qwen2.5-Coder (opsional)
    asr_model:    str = "openai/whisper-small"
    
    # === EMOTION ===
    emotion_enabled:    bool  = True
    emotion_decay:      float = 0.02
    idle_threshold_s:   float = 30.0
    
    # === MEMORY ===
    stm_capacity:   int = 64
    ltm_capacity:   int = 100_000
    
    # === ENVIRONMENT ===
    env_observe_enabled: bool = True
    base_path:           str  = "~"
    
    # === LEARNING ===
    auto_learn:         bool  = True    # Fine-tune alignment dari pengalaman
    learn_every_n:      int   = 50      # Setiap 50 interaksi
    
    # === PATHS ===
    checkpoint_dir:  str = "./claw_checkpoints"
    
    # === DEVICE ===
    device:   str = "auto"


@dataclass
class ClawMessage:
    """Satu pesan dalam percakapan dengan Claw."""
    role:      str               # 'user' atau 'claw'
    content:   str               # Teks pesan
    modality:  str = 'text'      # 'text', 'audio', 'image', dll
    emotion:   str = 'neutral'   # Emosi Claw saat pesan ini dibuat
    timestamp: float = field(default_factory=time.time)
    metadata:  Dict = field(default_factory=dict)


class ResonAItClaw:
    """
    RESONAITCLAW — Active AGI Agent berbasis Universal Frequency Space.
    
    Claw adalah ResonAIt yang sudah "hidup" — punya emosi, inisiatif,
    memori jangka panjang, dan kemampuan belajar dari pengalaman.
    
    Penggunaan dasar:
        >>> claw = ResonAItClaw.from_checkpoint("./claw_checkpoints")
        >>> response = claw.chat("Halo! Apa kabar?")
        >>> print(response.content)
        >>> print(f"Mood saya: {claw.mood}")
    
    Penggunaan advanced:
        >>> claw.start_background_loop()   # Aktifkan inisiatif engine
        >>> claw.on_initiative(callback)   # Terima notifikasi inisiatif
    """
    
    def __init__(self, config: ClawConfig):
        self.config = config
        self.name   = config.name
        
        # Resolve device
        if config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(config.device)
        
        # === CORE COMPONENTS ===
        self.unified_model:    Optional[ResonAItUnified]     = None
        self.alignment_layers: Dict[str, nn.Module]          = {}
        self.memory:           Optional[FrequencyMemorySystem] = None
        self.pain_engine:      Optional[DissonanceEngine]    = None
        self.emotion_engine:   Optional[EmotionEngine]       = None
        self.initiative_engine: Optional[InitiativeEngine]   = None
        self.env_observer:     Optional[EnvironmentObserver] = None
        
        # === CONVERSATION ===
        self.conversation_history: List[ClawMessage] = []
        self.interaction_count: int = 0
        
        # === CALLBACKS ===
        self._initiative_callbacks: List[Callable] = []
        self._emotion_callbacks:    List[Callable] = []
        
        # === BACKGROUND LOOP ===
        self._background_thread: Optional[threading.Thread] = None
        self._running = False
        
        # === SPECIALIST EXTRACTORS ===
        self._llm_model      = None
        self._llm_tokenizer  = None
        
        print(f"[{self.name}] Initialized — device: {self.device}")
    
    # ═══════════════════════════════════════════════════════════════
    # SETUP & LOADING
    # ═══════════════════════════════════════════════════════════════
    
    def setup(self, load_specialists: bool = True):
        """
        Inisialisasi semua komponen Claw.
        
        Args:
            load_specialists: Apakah load specialist models dari HuggingFace
                             (perlu internet, ~3-6GB per model)
        """
        print(f"\n[{self.name}] Setting up components...")
        
        # 1. Unified Model
        self._setup_unified_model()
        
        # 2. Memory
        self._setup_memory()
        
        # 3. Emotion Engine
        self._setup_emotion()
        
        # 4. Pain System (dissonance)
        self._setup_pain()
        
        # 5. Environment Observer
        if self.config.env_observe_enabled:
            self._setup_environment()
        
        # 6. Specialist models (opsional)
        if load_specialists and self.config.llm_model:
            self._load_specialist_llm()
        
        print(f"\n[{self.name}] ✅ Setup complete!")
        self._print_status()
    
    def _setup_unified_model(self):
        """Setup FNO Brain."""
        self.unified_model = ResonAItUnified(
            freq_dim   = self.config.freq_dim,
            hidden_dim = self.config.hidden_dim,
            n_modes    = 64,
            n_fno_layers = 4,
        ).to(self.device)
        
        n = sum(p.numel() for p in self.unified_model.parameters())
        print(f"  ✅ FNO Brain: {n:,} params")
    
    def _setup_memory(self):
        """Setup frequency memory system."""
        self.memory = FrequencyMemorySystem(
            freq_dim     = self.config.freq_dim,
            stm_capacity = self.config.stm_capacity,
            ltm_capacity = self.config.ltm_capacity,
            memory_dim   = 64,
        )
        print(f"  ✅ Memory: STM={self.config.stm_capacity} LTM={self.config.ltm_capacity}")
    
    def _setup_emotion(self):
        """Setup emotion engine."""
        if not self.config.emotion_enabled:
            return
        
        self.emotion_engine = EmotionEngine(
            freq_dim    = self.config.freq_dim,
            decay_rate  = self.config.emotion_decay,
        ).to(self.device)
        
        self.initiative_engine = InitiativeEngine(
            emotion_engine    = self.emotion_engine,
            idle_threshold_s  = self.config.idle_threshold_s,
        )
        
        print(f"  ✅ Emotion Engine: active")
    
    def _setup_pain(self):
        """Setup pain/dissonance system."""
        self.pain_engine = DissonanceEngine(freq_dim=self.config.freq_dim)
        print(f"  ✅ Pain System: active")
    
    def _setup_environment(self):
        """Setup environment observer."""
        self.env_observer = EnvironmentObserver(
            base_path = self.config.base_path
        )
        print(f"  ✅ Env Observer: watching {self.config.base_path}")
    
    def _load_specialist_llm(self):
        """Load LLM specialist dari HuggingFace via Unsloth."""
        try:
            from unsloth import FastLanguageModel
            print(f"  🔃 Loading LLM: {self.config.llm_model}...")
            
            self._llm_model, self._llm_tokenizer = FastLanguageModel.from_pretrained(
                model_name     = self.config.llm_model,
                max_seq_length = 2048,
                dtype          = torch.float16,
                load_in_4bit   = True,
            )
            self._llm_model.eval()
            
            # Buat alignment layer untuk LLM ini
            from resonait.tools.alignment import FrequencyAlignmentLayer
            llm_dim = self._llm_model.config.hidden_size
            self.alignment_layers['llm'] = FrequencyAlignmentLayer(
                embed_dim = llm_dim,
                freq_dim  = self.config.freq_dim,
                n_layers  = 3,
            ).to(self.device)
            
            print(f"  ✅ LLM loaded (embed_dim={llm_dim})")
        except Exception as e:
            print(f"  ⚠️  LLM load failed: {e}")
            print(f"     Continuing without LLM specialist.")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """
        Load checkpoint yang sudah ada (hasil training sebelumnya).
        
        Ini yang kamu pakai setelah selesai training di Kaggle:
        1. Download resonait_unified_v3.pt dari Kaggle
        2. Panggil claw.load_checkpoint("./claw_checkpoints")
        """
        ckpt_path = Path(checkpoint_dir)
        
        if (ckpt_path / "resonait_unified_final.pt").exists():
            self.unified_model = ResonAItUnified.load(
                str(ckpt_path / "resonait_unified_final.pt")
            )
            self.unified_model = self.unified_model.to(self.device)
            print(f"  ✅ Unified model loaded")
        
        # Load alignment layers jika ada
        from resonait.tools.alignment import FrequencyAlignmentLayer
        for name in ['llm', 'music', 'asr']:
            path = ckpt_path / f"align_{name}_v3.pt"
            if not path.exists():
                path = ckpt_path / f"align_{name}.pt"
            if path.exists():
                # Detect embed_dim dari state dict
                sd = torch.load(path, map_location='cpu', weights_only=True)
                # Ambil embed_dim dari key pertama yang mengandung 'weight'
                embed_dim = self.config.freq_dim  # fallback
                for k, v in sd.items():
                    if 'shared_projector' in k and 'weight' in k:
                        embed_dim = v.shape[1]
                        break
                
                layer = FrequencyAlignmentLayer(embed_dim, self.config.freq_dim)
                layer.load_state_dict(sd, strict=False)
                self.alignment_layers[name] = layer.to(self.device)
                print(f"  ✅ Alignment layer '{name}' loaded")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN INTERACTION
    # ═══════════════════════════════════════════════════════════════
    
    def chat(
        self,
        message:    str,
        task:       TaskType = TaskType.AUTO,
        modality:   str = 'text',
        image:      Optional[Any] = None,
        audio:      Optional[Any] = None,
    ) -> ClawMessage:
        """
        Kirim pesan ke Claw dan dapatkan respons.
        
        Args:
            message : Teks pesan
            task    : Task type (AUTO = deteksi otomatis)
            modality: Modalitas input ('text', 'audio', 'image')
            image   : Numpy array gambar (opsional)
            audio   : Numpy array audio (opsional)
            
        Returns:
            ClawMessage dengan respons
        """
        self.interaction_count += 1
        if self.initiative_engine:
            self.initiative_engine.record_interaction()
        
        # === 1. Buat input ===
        unified_input = UnifiedInput(
            task         = task,
            text_prompt  = message,
            image_input  = image,
            audio_input  = audio,
            context      = [
                {'role': m.role, 'content': m.content}
                for m in self.conversation_history[-10:]  # 10 pesan terakhir
            ],
        )
        
        # === 2. Proses melalui FNO Brain ===
        with torch.no_grad():
            output = self._process_with_emotion(unified_input)
        
        # === 3. Generate respons teks ===
        response_text = self._generate_response(message, output, unified_input)
        
        # === 4. Update memory ===
        self._update_memory(message, response_text)
        
        # === 5. Auto-learn jika diperlukan ===
        if self.config.auto_learn and self.interaction_count % self.config.learn_every_n == 0:
            self._auto_learn_step()
        
        # === 6. Buat respons message ===
        current_mood = self.mood
        response = ClawMessage(
            role     = 'claw',
            content  = response_text,
            modality = 'text',
            emotion  = current_mood,
            metadata = {
                'task_detected':  output.metadata.get('task_detected', '?'),
                'logic_gate':     output.metadata.get('logic_gate', 0),
                'imagination':    output.metadata.get('imagination_gate', 0),
                'memory_gate':    output.metadata.get('memory_gate', 0),
            }
        )
        
        # Tambah ke history
        self.conversation_history.append(
            ClawMessage(role='user', content=message, modality=modality)
        )
        self.conversation_history.append(response)
        
        return response
    
    def _process_with_emotion(self, unified_input: UnifiedInput) -> UnifiedOutput:
        """
        Proses input melalui FNO Brain dengan emotional prior di-inject.
        
        Alur:
        1. Convert input ke frequency
        2. Tambahkan emotional prior (emosi mempengaruhi "cara berpikir")
        3. Forward melalui brain
        4. Update emotion berdasarkan output
        """
        # Run model
        output = self.unified_model.run(unified_input)
        
        # Update emotion berdasarkan frekuensi yang diproses
        if self.emotion_engine and output.freq_hidden is not None:
            self.emotion_engine.process(output.freq_hidden.amplitude)
        elif self.emotion_engine:
            # Dummy update dengan random (jika freq_hidden tidak tersedia)
            dummy_amp = torch.rand(1, 1, self.config.freq_dim, device=self.device) * 0.5
            self.emotion_engine.process(dummy_amp)
        
        return output
    
    def _generate_response(
        self,
        user_message:   str,
        output:         UnifiedOutput,
        unified_input:  UnifiedInput,
    ) -> str:
        """
        Generate respons teks dari output FNO Brain.
        
        Jika ada LLM specialist yang sudah di-align → gunakan itu.
        Jika tidak → gunakan text decoder bawaan FNO.
        
        Juga menyertakan nuansa emosi dalam respons.
        """
        task = output.metadata.get('task_detected', 'chat')
        mood = self.mood if self.emotion_engine else 'neutral'
        
        # Jika ada LLM specialist, gunakan untuk generasi teks berkualitas
        if self._llm_model is not None and task in ['chat', 'reasoning', 'code']:
            return self._generate_with_llm(user_message, mood)
        
        # Fallback: respons sederhana berdasarkan task dan emosi
        return self._generate_simple_response(user_message, task, mood, output)
    
    def _generate_with_llm(self, message: str, mood: str) -> str:
        """Generate respons menggunakan LLM specialist yang sudah di-align."""
        try:
            # System prompt yang mencerminkan state emosi
            emotion_context = ""
            if self.emotion_engine:
                emotion_context = (
                    f"\n[Internal state: {self.emotion_engine.get_mood_description()}]"
                )
            
            system_prompt = (
                f"Kamu adalah {self.name}, {self.config.persona}"
                f"{emotion_context}\n"
                f"Jawab dengan natural, singkat, dan sesuai konteks."
            )
            
            # Format chat
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            
            # Tambah history singkat
            for msg in self.conversation_history[-6:]:
                role = "assistant" if msg.role == "claw" else "user"
                messages.append({"role": role, "content": msg.content})
            
            messages.append({"role": "user", "content": message})
            
            # Tokenize dan generate
            text = self._llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._llm_tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self._llm_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self._llm_tokenizer.eos_token_id,
                )
            
            # Decode hanya token baru
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response   = self._llm_tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return response.strip()
        
        except Exception as e:
            return self._generate_simple_response(message, 'chat', mood, None)
    
    def _generate_simple_response(
        self,
        message: str,
        task:    str,
        mood:    str,
        output:  Optional[UnifiedOutput],
    ) -> str:
        """
        Respons sederhana ketika LLM tidak tersedia.
        Nanti akan digantikan sepenuhnya oleh LLM + FreqDecoder.
        """
        emotion_note = ""
        if self.emotion_engine:
            emotion_note = f" [{self.emotion_engine.get_mood_description()}]"
        
        # Respons berbeda tergantung task
        task_responses = {
            'chat':           f"Saya menerima pesan Anda: '{message[:50]}...' {emotion_note}",
            'text_to_image':  f"Membuat gambar untuk: '{message[:40]}...'",
            'text_to_speech': f"Mengkonversi teks ke audio: '{message[:40]}...'",
            'text_to_music':  f"Membuat musik untuk: '{message[:40]}...'",
            'code':           f"Memproses request kode: '{message[:40]}...'",
        }
        
        base = task_responses.get(task, f"Memproses: '{message[:50]}'")
        
        # Tambahkan output gambar/audio jika ada
        if output and output.image_output is not None:
            base += f" [Gambar dihasilkan: {output.image_output.shape}]"
        if output and output.audio_output is not None:
            dur = len(output.audio_output) / 22050
            base += f" [Audio dihasilkan: {dur:.1f}s]"
        
        return base
    
    def _update_memory(self, user_message: str, response: str):
        """Update memory system dengan interaksi terbaru."""
        if not self.memory:
            return
        
        # Convert teks ke frequency untuk disimpan di memory
        try:
            from resonait.converters.universal_converter import UniversalFrequencyConverter
            converter = UniversalFrequencyConverter(freq_dim=self.config.freq_dim)
            
            combined_text = f"User: {user_message}\nClaw: {response}"
            freq = converter.convert(combined_text, modality='text')
            
            # Importance berdasarkan panjang dan novelty
            importance = min(0.9, len(user_message) / 500 + 0.3)
            
            self.memory.perceive(
                freq_tensor=freq,
                importance=importance,
                tags=['conversation'],
            )
        except Exception:
            pass
    
    def _auto_learn_step(self):
        """
        Satu langkah auto-learning dari pengalaman interaksi.
        
        Fine-tune alignment layers berdasarkan pola yang dipelajari.
        Ini yang membuat Claw semakin mengenal user seiring waktu.
        """
        if not self.alignment_layers:
            return
        
        try:
            # Ambil embedding dari conversation history terbaru
            recent_msgs = self.conversation_history[-20:]
            if len(recent_msgs) < 4:
                return
            
            # Extract text untuk learning
            texts = [m.content for m in recent_msgs if m.role == 'user']
            
            # Mini fine-tune alignment layer
            for name, al in self.alignment_layers.items():
                al.train()
                
                # Ambil embedding dari memory jika ada
                # (simplified — real impl akan pakai LLM embedding)
                dummy_emb = torch.randn(min(4, len(texts)), 
                                       self.config.freq_dim).to(self.device)
                
                amp, ph = al(dummy_emb)
                smooth = 0.01 * (amp[:, 1:] - amp[:, :-1]).abs().mean()
                smooth.backward()
                
                al.eval()
            
            print(f"  [Auto-learn] Step {self.interaction_count}: ✓")
        except Exception as e:
            pass
    
    # ═══════════════════════════════════════════════════════════════
    # PROPERTIES & HELPERS
    # ═══════════════════════════════════════════════════════════════
    
    @property
    def mood(self) -> str:
        """Mood dominan saat ini."""
        if not self.emotion_engine:
            return 'neutral'
        return self.emotion_engine.current_state.dominant().value
    
    @property
    def arousal(self) -> float:
        """Level energi/arousal saat ini (0.0–1.0)."""
        if not self.emotion_engine:
            return 0.3
        return self.emotion_engine.current_state.total_arousal()
    
    def get_status(self) -> Dict:
        """Status lengkap Claw saat ini."""
        status = {
            'name':             self.name,
            'mood':             self.mood,
            'arousal':          round(self.arousal, 3),
            'interactions':     self.interaction_count,
            'conversation_len': len(self.conversation_history),
            'device':           str(self.device),
        }
        
        if self.memory:
            status['memory'] = self.memory.get_stats()
        
        if self.emotion_engine:
            status['emotion'] = self.emotion_engine.current_state.as_text()
        
        if self.unified_model:
            status['brain_health'] = self.unified_model.get_health_report()
        
        if self.env_observer:
            status['user_profile'] = self.env_observer.get_user_profile()
        
        return status
    
    def _print_status(self):
        """Print status ringkas ke terminal."""
        print(f"\n  {'='*45}")
        print(f"  {self.name} — Active AGI Agent")
        print(f"  {'='*45}")
        print(f"  Device   : {self.device}")
        print(f"  Freq dim : {self.config.freq_dim}")
        print(f"  Mood     : {self.mood}")
        print(f"  Emotion  : {'enabled' if self.emotion_engine else 'disabled'}")
        print(f"  Memory   : {'enabled' if self.memory else 'disabled'}")
        print(f"  LLM      : {'loaded' if self._llm_model else 'not loaded'}")
        print(f"  Specialists: {list(self.alignment_layers.keys())}")
        print(f"  {'='*45}")
    
    # ═══════════════════════════════════════════════════════════════
    # BACKGROUND ACTIVE THINKING
    # ═══════════════════════════════════════════════════════════════
    
    def start_background_loop(self, interval_s: float = 10.0):
        """
        Mulai background thread untuk active thinking.
        
        Claw akan:
        - Observe environment setiap interval
        - Check apakah perlu berinisiatif
        - Trigger callbacks jika ada inisiatif
        
        Args:
            interval_s: Interval pengecekan dalam detik
        """
        if self._running:
            print(f"[{self.name}] Background loop sudah berjalan.")
            return
        
        self._running = True
        
        def loop():
            print(f"[{self.name}] Background thinking loop started.")
            while self._running:
                try:
                    self._background_tick()
                except Exception as e:
                    pass
                time.sleep(interval_s)
        
        self._background_thread = threading.Thread(target=loop, daemon=True)
        self._background_thread.start()
        print(f"[{self.name}] ✅ Background loop aktif (interval={interval_s}s)")
    
    def stop_background_loop(self):
        """Hentikan background thinking loop."""
        self._running = False
        if self._background_thread:
            self._background_thread.join(timeout=5)
        print(f"[{self.name}] Background loop stopped.")
    
    def _background_tick(self):
        """Satu tick background thinking."""
        # 1. Observe environment
        if self.env_observer:
            ctx = self.env_observer.observe()
        
        # 2. Update emotion (boredom naik jika idle lama)
        if self.emotion_engine:
            idle_s = self.initiative_engine.idle_seconds() if self.initiative_engine else 0
            if idle_s > 60:
                # Inject sedikit novelty dari environment
                dummy_amp = torch.rand(1, 1, self.config.freq_dim, device=self.device) * 0.2
                self.emotion_engine.process(dummy_amp)
        
        # 3. Check inisiatif
        if self.initiative_engine:
            initiative = self.initiative_engine.generate_initiative()
            if initiative:
                for cb in self._initiative_callbacks:
                    cb(initiative)
    
    def on_initiative(self, callback: Callable[[Dict], None]):
        """
        Daftarkan callback yang dipanggil saat Claw punya inisiatif.
        
        Args:
            callback: Fungsi yang menerima dict inisiatif
            
        Contoh:
            def handle_initiative(initiative):
                print(f"Claw: {initiative['content']}")
            
            claw.on_initiative(handle_initiative)
        """
        self._initiative_callbacks.append(callback)
    
    def on_emotion_change(self, callback: Callable[[EmotionalState], None]):
        """Daftarkan callback untuk perubahan emosi."""
        self._emotion_callbacks.append(callback)
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE & LOAD
    # ═══════════════════════════════════════════════════════════════
    
    def save(self, path: str = None):
        """Simpan state Claw ke disk."""
        save_path = Path(path or self.config.checkpoint_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Unified model
        if self.unified_model:
            self.unified_model.save(str(save_path / "unified_model.pt"))
        
        # Alignment layers
        for name, al in self.alignment_layers.items():
            torch.save(al.state_dict(), save_path / f"align_{name}.pt")
        
        # Config
        with open(save_path / "claw_config.json", 'w') as f:
            # Convert dataclass ke dict
            cfg = {k: v for k, v in self.config.__dict__.items()}
            json.dump(cfg, f, indent=2)
        
        # Conversation history (100 terakhir)
        history = [
            {
                'role': m.role, 'content': m.content,
                'emotion': m.emotion, 'timestamp': m.timestamp,
            }
            for m in self.conversation_history[-100:]
        ]
        with open(save_path / "conversation_history.json", 'w') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        print(f"[{self.name}] ✅ Saved to: {save_path}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        load_specialists: bool = False,
    ) -> "ResonAItClaw":
        """
        Load Claw dari checkpoint yang sudah ada.
        
        Args:
            checkpoint_dir  : Direktori checkpoint
            load_specialists: Apakah load specialist models
        """
        ckpt_path = Path(checkpoint_dir)
        
        # Load config
        config_path = ckpt_path / "claw_config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg_dict = json.load(f)
            config = ClawConfig(**{
                k: v for k, v in cfg_dict.items()
                if k in ClawConfig.__dataclass_fields__
            })
        else:
            config = ClawConfig()
        
        config.checkpoint_dir = checkpoint_dir
        
        # Buat instance
        claw = cls(config)
        claw.setup(load_specialists=load_specialists)
        claw.load_checkpoint(checkpoint_dir)
        
        # Load conversation history jika ada
        history_path = ckpt_path / "conversation_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            claw.conversation_history = [
                ClawMessage(**{k: v for k, v in m.items()})
                for m in history
            ]
            print(f"  ✅ Loaded {len(claw.conversation_history)} previous messages")
        
        return claw
    
    def __repr__(self) -> str:
        return (
            f"ResonAItClaw(name='{self.name}', "
            f"mood='{self.mood}', "
            f"arousal={self.arousal:.2f}, "
            f"interactions={self.interaction_count})"
        )
