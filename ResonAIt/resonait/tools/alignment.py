"""
resonait/tools/alignment.py
============================
THE "MAGIC" RETRAINING & ALIGNMENT TOOL

Modul terpenting untuk upgrade LLM privat (Llama-3, Mistral, dll)
ke format frekuensi ResonAIt.

Proses Alignment:
    LLM Lama (token space) → Pemetaan frekuensi → ResonAItBrain (frequency space)
    
Analoginya seperti "implant" neural:
    LLM yang sudah tahu banyak tentang dunia, tapi "hanya bisa membaca teks"
    di-upgrade agar bisa "melihat, mendengar, dan merasakan" melalui
    Universal Frequency Space.

Teknik yang digunakan:
    1. Knowledge Distillation     : LLM lama sebagai "guru", ResonAIt sebagai "murid"
    2. Frequency Alignment        : Mapping embedding space → frequency space
    3. LoRA Fine-tuning           : Efficient adaptation dengan sedikit parameter baru
    4. Multimodal Contrastive     : Pastikan teks, gambar, audio yang "sama" berdekatan
    5. Autonomous Data Sourcing   : Jika data kurang, generate data pendukung otomatis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import argparse
import json


# ============================================================
# KONFIGURASI ALIGNMENT
# ============================================================

@dataclass
class AlignmentConfig:
    """
    Konfigurasi lengkap untuk proses alignment LLM → ResonAIt.
    
    Semua parameter training dapat disesuaikan di sini.
    Buat dengan cara:
        config = AlignmentConfig(
            llm_model_name="meta-llama/Llama-3-8B",
            n_alignment_steps=1000,
        )
    """
    # === LLM Source ===
    llm_model_name:  str  = "mistralai/Mistral-7B-v0.1"  # Model HuggingFace atau path lokal
    llm_is_local:    bool = False  # True jika model ada di disk lokal
    
    # === ResonAIt Target ===
    freq_dim:     int = 512
    hidden_dim:   int = 256
    n_modes:      int = 64
    n_fno_layers: int = 4
    
    # === Training ===
    n_alignment_steps:   int   = 2000    # Jumlah step gradient untuk alignment
    batch_size:          int   = 4       # Ukuran batch
    learning_rate:       float = 1e-4    # Learning rate
    weight_decay:        float = 0.01    # L2 regularization
    warmup_steps:        int   = 100     # Warmup LR schedule
    
    # === LoRA (Parameter-Efficient Fine-Tuning) ===
    use_lora:    bool = True   # Gunakan LoRA untuk efisiensi
    lora_rank:   int  = 16     # Rank LoRA (lebih tinggi = lebih expressive, lebih berat)
    lora_alpha:  int  = 32     # Scaling factor LoRA
    lora_dropout: float = 0.1  # Dropout dalam LoRA
    
    # === Loss Weights ===
    # Seberapa besar kontribusi setiap komponen loss
    distillation_weight:   float = 1.0  # Knowledge distillation dari LLM guru
    alignment_weight:      float = 2.0  # Alignment embedding → frekuensi (prioritas utama)
    contrastive_weight:    float = 0.5  # Multimodal contrastive learning
    reconstruction_weight: float = 0.3  # Rekonstruksi teks dari frekuensi
    
    # === Data ===
    data_path:       Optional[str] = None   # Path ke dataset (opsional)
    auto_expand:     bool = True            # Aktifkan Autonomous Data Sourcing
    auto_expand_strategy: str = "simulate" # "simulate", "search", atau "generate"
    
    # === Output ===
    output_dir:    str  = "./resonait_aligned"  # Direktori output
    save_every:    int  = 500                    # Simpan checkpoint setiap N steps
    eval_every:    int  = 100                    # Evaluasi setiap N steps
    
    # === Device ===
    device:    str  = "auto"   # "auto", "cpu", "cuda", "mps"
    precision: str  = "fp16"   # "fp32", "fp16", "bf16"


# ============================================================
# FREQUENCY ALIGNMENT LAYER
# ============================================================

class FrequencyAlignmentLayer(nn.Module):
    """
    Layer yang memetakan embedding space LLM ke Universal Frequency Space.
    
    Ini adalah "jembatan" yang menghubungkan representasi token LLM lama
    dengan domain frekuensi baru ResonAIt.
    
    Secara matematis:
        f: R^(embed_dim) → R^(freq_dim) × R^(freq_dim)
        f(e) = (A(e), φ(e))  [amplitude, phase]
    
    Di mana A dan φ dipelajari oleh jaringan neural.
    
    Args:
        embed_dim (int): Dimensi embedding LLM sumber
        freq_dim  (int): Dimensi frekuensi target ResonAIt
        n_layers  (int): Kedalaman proyeksi (lebih dalam = lebih ekspresif)
    """
    
    def __init__(
        self,
        embed_dim: int,
        freq_dim:  int,
        n_layers:  int = 3,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.freq_dim  = freq_dim
        
        # === PROYEKTOR BERSAMA ===
        # Lapisan awal yang dipakai bersama sebelum memisah ke amplitude dan phase
        shared_layers = []
        current_dim   = embed_dim
        
        for i in range(n_layers - 1):
            # Reduksi dimensi secara gradual
            next_dim = max(freq_dim, current_dim // 2)
            shared_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.LayerNorm(next_dim),
                nn.GELU(),
            ])
            current_dim = next_dim
        
        self.shared_projector = nn.Sequential(*shared_layers)
        
        # === PROYEKTOR AMPLITUDE ===
        # Menghasilkan amplitude yang selalu positif (Softplus memastikan ini)
        self.amplitude_head = nn.Sequential(
            nn.Linear(current_dim, freq_dim),
            nn.Softplus(),  # Selalu positif — amplitude tidak boleh negatif
        )
        
        # === PROYEKTOR PHASE ===
        # Menghasilkan phase dalam range (-π, π)
        self.phase_head = nn.Sequential(
            nn.Linear(current_dim, freq_dim),
            nn.Tanh(),      # Range (-1, 1) × π → (-π, π)
        )
        
        # Scaling untuk phase
        self.phase_scale = nn.Parameter(torch.tensor(np.pi))
    
    def forward(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Peta embedding LLM ke amplitude dan phase frekuensi.
        
        Args:
            embeddings: Embedding dari LLM. Shape: (batch, seq_len, embed_dim)
                        atau (batch, embed_dim) jika sudah di-pool
            
        Returns:
            Tuple (amplitude, phase), masing-masing shape (batch, freq_dim)
        """
        # Proyeksi bersama
        shared = self.shared_projector(embeddings)
        
        # Proyeksi ke amplitude (selalu positif)
        amplitude = self.amplitude_head(shared)
        
        # Proyeksi ke phase (dalam radian)
        phase = self.phase_head(shared) * self.phase_scale
        
        return amplitude, phase


# ============================================================
# KELAS UTAMA: LLMAlignmentTool
# ============================================================

class LLMAlignmentTool:
    """
    Tool utama untuk meng-upgrade LLM privat ke format ResonAIt.
    
    Proses alignment berlangsung dalam beberapa fase:
    
    Fase 1 - Analisis LLM Sumber:
        Ambil embedding layer dan representasi internal LLM lama.
        
    Fase 2 - Inisialisasi FrequencyAlignmentLayer:
        Buat layer pemetaan dari embedding space ke frequency space.
        
    Fase 3 - Knowledge Distillation Training:
        Latih ResonAIt untuk "meniru" LLM lama dalam domain frekuensi.
        LLM lama = teacher, ResonAIt = student.
        
    Fase 4 - Multimodal Contrastive Alignment:
        Pastikan teks, gambar, audio yang semantically sama
        menghasilkan frekuensi yang berdekatan di frequency space.
        
    Fase 5 - Fine-tuning dan Evaluasi:
        Fine-tune seluruh sistem dan evaluasi kualitas alignment.
    
    Penggunaan:
        >>> tool = LLMAlignmentTool(config)
        >>> tool.load_source_llm("meta-llama/Llama-3-8B")
        >>> tool.align(training_data)
        >>> tool.save()
    """
    
    def __init__(self, config: AlignmentConfig):
        self.config = config
        
        # Resolve device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"[AlignmentTool] Device: {self.device}")
        
        # Komponen utama — diinisialisasi saat load_source_llm() dipanggil
        self.source_llm         = None
        self.source_tokenizer   = None
        self.source_embed_dim   = None
        self.alignment_layer    = None
        self.resonait_brain     = None
        self.optimizer          = None
        self.scheduler          = None
        
        # Training history
        self.training_history: List[Dict] = []
    
    def load_source_llm(self, model_name_or_path: Optional[str] = None):
        """
        Muat LLM sumber yang akan di-align ke ResonAIt.
        
        Mendukung model HuggingFace manapun (Llama, Mistral, Gemma, dll).
        Hanya memuat model dalam mode "embedding extraction" —
        tidak perlu GPU besar karena kita tidak generate teks.
        
        Args:
            model_name_or_path: Nama model HuggingFace atau path lokal.
                               Jika None, gunakan config.llm_model_name
        """
        from transformers import AutoModel, AutoTokenizer
        
        model_id = model_name_or_path or self.config.llm_model_name
        print(f"\n[AlignmentTool] ⚙ Loading LLM sumber: {model_id}")
        print("[AlignmentTool] (Ini mungkin memerlukan beberapa menit...)")
        
        # Muat tokenizer
        self.source_tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        # Pastikan ada padding token
        if self.source_tokenizer.pad_token is None:
            self.source_tokenizer.pad_token = self.source_tokenizer.eos_token
        
        # Muat model dalam precision yang sesuai
        # Kita gunakan model full untuk mendapatkan embedding berkualitas tinggi
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        load_dtype = dtype_map.get(self.config.precision, torch.float16)
        
        # Muat hanya untuk inference (tidak perlu gradient)
        self.source_llm = AutoModel.from_pretrained(
            model_id,
            torch_dtype=load_dtype,
            device_map="auto",          # Distribusikan otomatis ke GPU yang tersedia
            trust_remote_code=True,
            output_hidden_states=True,  # Kita butuh hidden states untuk distillation
        )
        self.source_llm.eval()  # Mode inference
        
        # Freeze semua parameter LLM sumber
        # Kita TIDAK melatih ulang LLM lama — kita hanya mengekstrak pengetahuannya
        for param in self.source_llm.parameters():
            param.requires_grad = False
        
        # Dapatkan dimensi embedding
        # Berbeda untuk setiap model — kita ambil secara dinamis
        config_obj = self.source_llm.config
        self.source_embed_dim = getattr(
            config_obj, 'hidden_size',
            getattr(config_obj, 'd_model', 768)
        )
        
        print(f"[AlignmentTool] ✓ LLM sumber dimuat!")
        print(f"[AlignmentTool]   Embed dim: {self.source_embed_dim}")
        print(f"[AlignmentTool]   Parameter: {sum(p.numel() for p in self.source_llm.parameters()):,}")
        
        # Inisialisasi komponen-komponen yang bergantung pada embed_dim
        self._initialize_alignment_components()
    
    def _initialize_alignment_components(self):
        """
        Inisialisasi semua komponen alignment setelah embed_dim diketahui.
        
        Urutan inisialisasi:
        1. FrequencyAlignmentLayer: Pemetaan embedding → frekuensi
        2. ResonAItBrain: Otak target yang akan menerima pengetahuan
        3. Optimizer dan Scheduler
        """
        from resonait.core.brain import ResonAItBrain
        
        print("\n[AlignmentTool] ⚙ Menginisialisasi komponen alignment...")
        
        # === 1. FREQUENCY ALIGNMENT LAYER ===
        self.alignment_layer = FrequencyAlignmentLayer(
            embed_dim=self.source_embed_dim,
            freq_dim=self.config.freq_dim,
            n_layers=3,
        ).to(self.device)
        
        # === 2. RESONAIT BRAIN ===
        self.resonait_brain = ResonAItBrain(
            freq_dim=self.config.freq_dim,
            hidden_dim=self.config.hidden_dim,
            n_modes=self.config.n_modes,
            n_fno_layers=self.config.n_fno_layers,
        ).to(self.device)
        
        # === 3. OPTIMIZER ===
        # Hanya train alignment_layer dan resonait_brain (bukan source LLM)
        trainable_params = (
            list(self.alignment_layer.parameters()) +
            list(self.resonait_brain.parameters())
        )
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Cosine annealing dengan warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.n_alignment_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        total_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
        print(f"[AlignmentTool] ✓ Alignment layer: {sum(p.numel() for p in self.alignment_layer.parameters()):,} params")
        print(f"[AlignmentTool] ✓ ResonAIt Brain: {sum(p.numel() for p in self.resonait_brain.parameters()):,} params")
        print(f"[AlignmentTool] ✓ Total trainable: {total_trainable:,} params")
    
    def extract_llm_embeddings(
        self,
        texts: List[str],
        layer_index: int = -1
    ) -> torch.Tensor:
        """
        Ekstrak hidden state embeddings dari LLM sumber.
        
        Mengambil representasi internal LLM pada layer tertentu.
        Layer terakhir biasanya paling kaya secara semantik.
        
        Args:
            texts      : List string yang akan diembed
            layer_index: Index layer yang akan diambil (-1 = layer terakhir)
            
        Returns:
            Embedding tensor. Shape: (batch, embed_dim)
        """
        # Tokenisasi dengan padding
        inputs = self.source_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        # Ekstrak embedding tanpa gradient
        with torch.no_grad():
            outputs = self.source_llm(**inputs, output_hidden_states=True)
        
        # Ambil hidden states dari layer yang ditentukan
        # hidden_states adalah tuple: (embedding_layer, layer_1, ..., layer_N)
        hidden_states = outputs.hidden_states
        
        if layer_index == -1:
            # Layer terakhir
            embeddings = hidden_states[-1]
        else:
            embeddings = hidden_states[layer_index]
        
        # Shape: (batch, seq_len, embed_dim)
        # Pool over sequence length menggunakan mean pooling
        # (dengan memperhatikan padding mask)
        attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
        sum_embeddings = (embeddings * attention_mask).sum(dim=1)
        count_tokens   = attention_mask.sum(dim=1)
        mean_embedding = sum_embeddings / (count_tokens + 1e-8)
        # Shape: (batch, embed_dim)
        
        return mean_embedding.to(self.config.precision == "fp32" and torch.float32 or torch.float32)
    
    def compute_alignment_loss(
        self,
        texts:           List[str],
        image_data:      Optional[List[Any]] = None,
        audio_data:      Optional[List[Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Hitung total alignment loss untuk satu batch.
        
        Komponen loss:
        1. Alignment Loss   : Jarak antara embedding LLM dan frekuensi yang dihasilkan
        2. Contrastive Loss : Pastikan multimodal yang sama berdekatan di freq space
        3. Reconstruction   : Seberapa baik kita bisa "mengingat" embedding asli
        
        Args:
            texts     : Batch teks input
            image_data: Data gambar pendukung (opsional)
            audio_data: Data audio pendukung (opsional)
            
        Returns:
            Dict berisi setiap komponen loss dan total loss
        """
        from resonait.core.frequency_space import FrequencyTensor, Modality
        from resonait.converters.text_converter import TextConverter
        
        text_converter = TextConverter(freq_dim=self.config.freq_dim)
        
        # === Step 1: Dapatkan embedding dari LLM sumber (teacher) ===
        teacher_embeddings = self.extract_llm_embeddings(texts)
        # Shape: (batch, embed_dim)
        
        # === Step 2: Petakan embedding ke domain frekuensi ===
        student_amplitude, student_phase = self.alignment_layer(
            teacher_embeddings.to(self.device)
        )
        # Shape: (batch, freq_dim)
        
        # === Step 3: Hitung Alignment Loss ===
        # Kita ingin amplitude yang dihasilkan "merepresentasikan" embedding asli.
        # Gunakan reconstruction loss: bisa kita rekonstruksi embedding dari frekuensi?
        
        # Buat representasi kompleks dari hasil alignment
        complex_repr = torch.complex(
            student_amplitude * torch.cos(student_phase),
            student_amplitude * torch.sin(student_phase),
        )
        
        # IFFT untuk rekonstruksi
        reconstructed_signal = torch.fft.irfft(complex_repr, n=self.source_embed_dim, norm="ortho")
        
        # Jika dimensi tidak sama, interpolasi
        if reconstructed_signal.shape[-1] != teacher_embeddings.shape[-1]:
            reconstructed_signal = F.interpolate(
                reconstructed_signal.unsqueeze(1),
                size=teacher_embeddings.shape[-1],
                mode='linear',
                align_corners=False
            ).squeeze(1)
        
        # Reconstruction loss: seberapa mirip rekonstruksi dengan embedding asli
        alignment_loss = F.mse_loss(
            reconstructed_signal,
            teacher_embeddings.detach()  # Detach — kita tidak ingin gradient mengalir ke LLM teacher
        )
        
        # === Step 4: Contrastive Loss (jika multimodal data tersedia) ===
        contrastive_loss = torch.tensor(0.0, device=self.device)
        
        if image_data is not None:
            from resonait.converters.image_converter import ImageConverter
            img_converter = ImageConverter(freq_dim=self.config.freq_dim)
            
            # Ekstrak frekuensi dari gambar
            img_amplitudes = []
            for img in image_data:
                img_freq = img_converter.to_frequency_tensor(img)
                img_amplitudes.append(img_freq.amplitude[:, 0, :])  # (1, freq_dim)
            
            img_amp_batch = torch.cat(img_amplitudes, dim=0).to(self.device)
            
            # Contrastive: teks dan gambar yang "sama" harus berdekatan di freq space
            # Gunakan cosine similarity
            text_norm = F.normalize(student_amplitude, dim=-1)
            img_norm  = F.normalize(img_amp_batch, dim=-1)
            
            # Positive pairs: diagonal (teks ke-i harus dekat dengan gambar ke-i)
            similarity_matrix = torch.matmul(text_norm, img_norm.T)
            
            # InfoNCE loss
            temperature = 0.07
            labels      = torch.arange(len(texts), device=self.device)
            contrastive_loss = (
                F.cross_entropy(similarity_matrix / temperature, labels) +
                F.cross_entropy(similarity_matrix.T / temperature, labels)
            ) / 2
        
        # === Step 5: Total Loss ===
        total_loss = (
            self.config.alignment_weight  * alignment_loss +
            self.config.contrastive_weight * contrastive_loss
        )
        
        return {
            "total":       total_loss,
            "alignment":   alignment_loss,
            "contrastive": contrastive_loss,
        }
    
    def align(
        self,
        training_texts: List[str],
        training_images: Optional[List] = None,
        training_audios: Optional[List] = None,
    ):
        """
        FUNGSI UTAMA: Jalankan proses alignment penuh.
        
        Ini adalah "magic" dari ResonAIt — mengambil LLM yang sudah dilatih
        dan mentransfer pengetahuannya ke Universal Frequency Space.
        
        Args:
            training_texts : List teks untuk training
            training_images: List gambar pendukung (opsional)
            training_audios: List audio pendukung (opsional)
        """
        if self.source_llm is None:
            raise RuntimeError(
                "LLM sumber belum dimuat. Panggil load_source_llm() terlebih dahulu."
            )
        
        print(f"\n{'='*60}")
        print(f"  RESONAIT ALIGNMENT TOOL — MULAI TRAINING")
        print(f"{'='*60}")
        print(f"  LLM Sumber  : {self.config.llm_model_name}")
        print(f"  Data texts  : {len(training_texts)} sampel")
        print(f"  Steps       : {self.config.n_alignment_steps}")
        print(f"  Device      : {self.device}")
        print(f"{'='*60}\n")
        
        # === Autonomous Data Sourcing ===
        # Jika tidak ada data gambar/audio, buat secara otomatis
        if training_images is None and self.config.auto_expand:
            print("[AlignmentTool] 🔍 Autonomous Data Sourcing: membuat data gambar pendukung...")
            from resonait.converters.universal_converter import UniversalFrequencyConverter
            ufc = UniversalFrequencyConverter(freq_dim=self.config.freq_dim)
            
            # Generate supporting image data dari teks
            training_images = [
                ufc._create_supporting_image(text, self.config.auto_expand_strategy)
                for text in training_texts
            ]
            print(f"[AlignmentTool] ✓ {len(training_images)} gambar sintetis dibuat.")
        
        # Set ke training mode
        self.alignment_layer.train()
        self.resonait_brain.train()
        
        # Output directory
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # === TRAINING LOOP ===
        n_samples   = len(training_texts)
        batch_size  = self.config.batch_size
        
        for step in range(self.config.n_alignment_steps):
            # Sample batch secara acak
            indices     = np.random.choice(n_samples, size=batch_size, replace=(n_samples < batch_size))
            batch_texts = [training_texts[i] for i in indices]
            batch_images = [training_images[i] for i in indices] if training_images else None
            
            # Hitung loss
            self.optimizer.zero_grad()
            losses = self.compute_alignment_loss(
                texts=batch_texts,
                image_data=batch_images,
                audio_data=None,
            )
            
            # Backpropagation
            losses["total"].backward()
            
            # Gradient clipping untuk stabilitas
            torch.nn.utils.clip_grad_norm_(
                list(self.alignment_layer.parameters()) +
                list(self.resonait_brain.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Catat history
            step_log = {
                "step":        step,
                "loss_total":  losses["total"].item(),
                "loss_align":  losses["alignment"].item(),
                "loss_contrastive": losses["contrastive"].item(),
                "lr":          self.scheduler.get_last_lr()[0],
            }
            self.training_history.append(step_log)
            
            # Progress logging
            if step % 50 == 0 or step == self.config.n_alignment_steps - 1:
                print(
                    f"Step [{step:4d}/{self.config.n_alignment_steps}] | "
                    f"Total Loss: {step_log['loss_total']:.4f} | "
                    f"Align: {step_log['loss_align']:.4f} | "
                    f"Contrast: {step_log['loss_contrastive']:.4f} | "
                    f"LR: {step_log['lr']:.2e}"
                )
            
            # Simpan checkpoint
            if (step + 1) % self.config.save_every == 0:
                self._save_checkpoint(step, output_path)
        
        print(f"\n[AlignmentTool] ✓ Training selesai!")
        self._save_final(output_path)
    
    def _save_checkpoint(self, step: int, output_path: Path):
        """Simpan checkpoint intermediate."""
        ckpt = {
            "step":                  step,
            "alignment_layer_state": self.alignment_layer.state_dict(),
            "resonait_brain_state":  self.resonait_brain.state_dict(),
            "optimizer_state":       self.optimizer.state_dict(),
            "config":                self.config.__dict__,
            "training_history":      self.training_history,
        }
        path = output_path / f"checkpoint_step_{step}.pt"
        torch.save(ckpt, path)
        print(f"[AlignmentTool] 💾 Checkpoint disimpan: {path}")
    
    def _save_final(self, output_path: Path):
        """Simpan model final yang sudah di-align."""
        # Simpan alignment layer
        torch.save(
            self.alignment_layer.state_dict(),
            output_path / "alignment_layer_final.pt"
        )
        
        # Simpan ResonAIt brain yang sudah ter-align
        self.resonait_brain.save(str(output_path / "resonait_brain_aligned.pt"))
        
        # Simpan config
        with open(output_path / "alignment_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Simpan training history
        with open(output_path / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n[AlignmentTool] ✅ Alignment selesai! Semua file tersimpan di: {output_path}")
        print(f"[AlignmentTool]    - alignment_layer_final.pt")
        print(f"[AlignmentTool]    - resonait_brain_aligned.pt")
        print(f"[AlignmentTool]    - alignment_config.json")
        print(f"[AlignmentTool]    - training_history.json")
    
    @classmethod
    def load_aligned(cls, output_dir: str) -> "LLMAlignmentTool":
        """
        Muat hasil alignment yang sudah selesai.
        
        Args:
            output_dir: Direktori output yang berisi hasil alignment
            
        Returns:
            LLMAlignmentTool dengan model yang sudah ter-align
        """
        from resonait.core.brain import ResonAItBrain
        
        output_path = Path(output_dir)
        
        # Load config
        with open(output_path / "alignment_config.json") as f:
            config_dict = json.load(f)
        config = AlignmentConfig(**{
            k: v for k, v in config_dict.items()
            if k in AlignmentConfig.__dataclass_fields__
        })
        
        tool = cls(config)
        tool._initialize_alignment_components()
        
        # Load weights
        tool.alignment_layer.load_state_dict(
            torch.load(output_path / "alignment_layer_final.pt", map_location="cpu")
        )
        tool.resonait_brain = ResonAItBrain.load(
            str(output_path / "resonait_brain_aligned.pt")
        )
        
        print(f"[AlignmentTool] ✓ Model ter-align dimuat dari: {output_dir}")
        return tool


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main_cli():
    """
    CLI interface untuk menjalankan alignment dari terminal.
    
    Contoh:
        resonait-align --model mistralai/Mistral-7B-v0.1 \\
                       --data data/texts.json \\
                       --steps 2000 \\
                       --output ./my_resonait
    """
    parser = argparse.ArgumentParser(
        description="ResonAIt Alignment Tool — Upgrade LLM ke Universal Frequency Space"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Nama atau path LLM sumber (e.g., mistralai/Mistral-7B-v0.1)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path ke file teks (JSON list atau plaintext)")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Jumlah alignment steps")
    parser.add_argument("--output", type=str, default="./resonait_aligned",
                        help="Direktori output")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--no-auto-expand", action="store_true",
                        help="Nonaktifkan Autonomous Data Sourcing")
    
    args = parser.parse_args()
    
    # Load training data
    data_path = Path(args.data)
    if data_path.suffix == ".json":
        with open(data_path) as f:
            training_texts = json.load(f)
    else:
        with open(data_path) as f:
            training_texts = [line.strip() for line in f if line.strip()]
    
    print(f"[CLI] {len(training_texts)} teks dimuat dari {args.data}")
    
    # Buat config
    config = AlignmentConfig(
        llm_model_name=args.model,
        n_alignment_steps=args.steps,
        output_dir=args.output,
        device=args.device,
        auto_expand=not args.no_auto_expand,
    )
    
    # Jalankan alignment
    tool = LLMAlignmentTool(config)
    tool.load_source_llm()
    tool.align(training_texts)


if __name__ == "__main__":
    main_cli()
