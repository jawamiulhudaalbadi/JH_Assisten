"""Router model Gemini yang belajar sendiri.

Tugasnya: memilih model yang paling baru, sehat, DAN cukup cepat, berdasarkan
pengukuran nyata dari aplikasi (bukan daftar tetap).

  * Kecepatan  : rata-rata waktu-token-pertama (EWMA) dicatat tiap jawaban.
                 Model yang kalah balapan (terlalu lambat) dicatat lewat
                 record_slow() sehingga diturunkan bila terjadi berulang.
  * Kesehatan  : gagal (503/timeout/429) -> cooldown bertingkat 60s, 2m, 4m ...
                 maksimal 15 menit. 404 -> dilewati 6 jam.
  * Pulih      : setelah cooldown habis model dicoba lagi; bila berhasil, ia
                 otomatis kembali ke posisi semula.

Aman dipakai dari banyak thread. Statistik disimpan di memori proses.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


SLOW_SECONDS = _env_float("ROUTER_SLOW_SECONDS", 8.0)
BASE_COOLDOWN = _env_float("ROUTER_BASE_COOLDOWN", 60.0)
MAX_COOLDOWN = _env_float("ROUTER_MAX_COOLDOWN", 900.0)
NOT_FOUND_COOLDOWN = _env_float("ROUTER_NOT_FOUND_COOLDOWN", 6 * 3600.0)
EWMA_ALPHA = 0.4  # bobot pengukuran terbaru
INCLUDE_PRO = os.getenv("GEMINI_INCLUDE_PRO", "").strip().lower() in {"1", "true", "yes"}

# Urutan manual (opsional), contoh: GEMINI_CHAT_MODELS="gemini-3.1-flash-lite,gemini-3.8-flash"
# Bila diisi, HANYA model ini yang dipakai, dengan prioritas sesuai urutan tulis.
MANUAL_ORDER = [
    n.strip().removeprefix("models/")
    for n in os.getenv("GEMINI_CHAT_MODELS", "").split(",")
    if n.strip()
]

_MODEL_RE = re.compile(r"^gemini-(\d+(?:\.\d+)?)-(flash-lite|flash|pro)$")


def _normalize(name: str) -> str:
    return str(name).removeprefix("models/")


def _tier_and_version(name: str) -> tuple[int, float]:
    """Kecil = lebih diutamakan. flash > flash-lite > tak dikenal > pro."""
    match = _MODEL_RE.match(name)
    if not match:
        # gemma, gemini-4-argon, dll: stabilitas belum diketahui -> setelah yang dikenal
        return (3, 0.0)
    version = float(match.group(1))
    family = match.group(2)
    tier = {"flash": 0, "flash-lite": 1, "pro": 4}[family]
    return (tier, -version)


@dataclass
class _Stats:
    ewma_ttft: float | None = None
    samples: int = 0
    failures: int = 0  # kegagalan beruntun
    cooldown_until: float = 0.0
    not_found: bool = False
    thinking_unsupported: bool = False
    last_error: str = ""
    successes: int = 0
    total_failures: int = 0
    slow_losses: int = 0


@dataclass
class ModelRouter:
    clock: Callable[[], float] = time.monotonic
    _stats: dict[str, _Stats] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: str = ""

    # ---- pencatatan -------------------------------------------------------
    def _get(self, name: str) -> _Stats:
        return self._stats.setdefault(name, _Stats())

    def _update_ewma(self, stat: _Stats, seconds: float) -> None:
        stat.samples += 1
        stat.ewma_ttft = (
            seconds
            if stat.ewma_ttft is None
            else EWMA_ALPHA * seconds + (1 - EWMA_ALPHA) * stat.ewma_ttft
        )

    def record_success(self, model: str, seconds: float) -> None:
        name = _normalize(model)
        if not name:
            return
        with self._lock:
            stat = self._get(name)
            stat.failures = 0
            stat.cooldown_until = 0.0
            stat.not_found = False
            stat.successes += 1
            stat.slow_losses = 0
            self._update_ewma(stat, seconds)
            self.last_used = name

    def record_slow(self, model: str, seconds: float) -> None:
        """Model kalah balapan: catat sebagai pengukuran lambat (bukan gagal)."""
        name = _normalize(model)
        if not name:
            return
        with self._lock:
            stat = self._get(name)
            stat.slow_losses += 1
            self._update_ewma(stat, seconds)

    def record_failure(self, model: str, error: BaseException | str = "") -> str:
        """Catat kegagalan. Mengembalikan jenis: notfound|thinking|transient."""
        name = _normalize(model)
        text = str(error).casefold()
        with self._lock:
            stat = self._get(name)
            stat.last_error = str(error)[:200]
            if "thinking" in text and ("400" in text or "invalid" in text):
                # Bukan model rusak: hanya parameter berpikir yang tidak didukung.
                stat.thinking_unsupported = True
                return "thinking"
            stat.total_failures += 1
            if any(m in text for m in ("404", "not_found", "no longer available")):
                stat.not_found = True
                stat.cooldown_until = self.clock() + NOT_FOUND_COOLDOWN
                return "notfound"
            stat.failures += 1
            delay = min(BASE_COOLDOWN * (2 ** (stat.failures - 1)), MAX_COOLDOWN)
            stat.cooldown_until = self.clock() + delay
            return "transient"

    def mark_thinking_unsupported(self, model: str) -> None:
        with self._lock:
            self._get(_normalize(model)).thinking_unsupported = True

    def thinking_supported(self, model: str) -> bool:
        with self._lock:
            return not self._get(_normalize(model)).thinking_unsupported

    def expected_ttft(self, model: str) -> float | None:
        with self._lock:
            return self._get(_normalize(model)).ewma_ttft

    # ---- pengurutan -------------------------------------------------------
    def rank(self, models: Iterable[str]) -> list[str]:
        """Urutan terbaik: sehat+cepat dulu, lalu lambat, lalu yang sedang cooldown."""
        now = self.clock()
        names = [_normalize(m) for m in dict.fromkeys(models)]
        if MANUAL_ORDER:
            names = MANUAL_ORDER[:]  # tetap dicoba walau tak muncul di daftar API
        elif not INCLUDE_PRO:
            non_pro = [n for n in names if "pro" not in n.split("-")]
            names = non_pro or names

        healthy: list[tuple] = []
        slow: list[tuple] = []
        cooling: list[tuple] = []
        dead: list[tuple] = []
        with self._lock:
            for name in names:
                stat = self._get(name)
                tier, ver = _tier_and_version(name)
                if MANUAL_ORDER:
                    tier, ver = MANUAL_ORDER.index(name), 0.0
                if stat.not_found and stat.cooldown_until > now:
                    dead.append((stat.cooldown_until, name))
                elif stat.cooldown_until > now:
                    cooling.append((stat.cooldown_until, tier, ver, name))
                elif stat.ewma_ttft is not None and stat.ewma_ttft > SLOW_SECONDS:
                    # Lambat: yang paling cepat di antara yang lambat dulu.
                    slow.append((stat.ewma_ttft, tier, ver, name))
                else:
                    healthy.append((tier, ver, name))

        healthy.sort()
        slow.sort()
        cooling.sort()  # yang paling cepat selesai cooldown dulu
        dead.sort()
        ordered = (
            [h[-1] for h in healthy]
            + [s[-1] for s in slow]
            + [c[-1] for c in cooling]
            + [d[-1] for d in dead]
        )
        # Jangan sia-siakan percobaan pada model 404 bila masih ada alternatif lain.
        alive = [n for n in ordered if n not in {d[-1] for d in dead}]
        return alive or ordered

    def first_token_timeout(self, model: str, default: float) -> float:
        """Batas tunggu token pertama: 3x rata-rata, dalam rentang 8s..default."""
        with self._lock:
            ewma = self._get(_normalize(model)).ewma_ttft
        if ewma is None:
            return default
        return max(8.0, min(default, ewma * 3.0))

    # ---- pemanasan --------------------------------------------------------
    def warm_up(
        self,
        models: Sequence[str],
        probe: Callable[[str], float],
        max_models: int = 4,
        deadline: float = 20.0,
    ) -> None:
        """Ukur beberapa model teratas secara paralel dengan prompt kecil.

        probe(model) harus mengembalikan detik sampai token pertama, atau
        melempar exception bila gagal.
        """
        targets = self.rank(models)[:max_models]

        def run(name: str) -> None:
            try:
                self.record_success(name, probe(name))
            except Exception as error:  # noqa: BLE001
                self.record_failure(name, error)

        threads = [threading.Thread(target=run, args=(n,), daemon=True) for n in targets]
        for t in threads:
            t.start()
        end = self.clock() + deadline
        for t in threads:
            t.join(max(0.0, end - self.clock()))

    # ---- inspeksi ---------------------------------------------------------
    def snapshot(self) -> list[dict]:
        now = self.clock()
        with self._lock:
            return [
                {
                    "model": name,
                    "ttft": None if s.ewma_ttft is None else round(s.ewma_ttft, 2),
                    "samples": s.samples,
                    "cooldown_sisa": max(0, round(s.cooldown_until - now)),
                    "gagal_beruntun": s.failures,
                    "kalah_balapan": s.slow_losses,
                    "last_error": s.last_error,
                }
                for name, s in sorted(self._stats.items())
            ]


router = ModelRouter()