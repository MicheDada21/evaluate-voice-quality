"""Shared voice toolkit: audio decoding, Scribe transcripts, speech regions, voiceprints and pitch.

Stack: ffmpeg (decoding), ElevenLabs Scribe v2 through the `elevenlabs` CLI (words, timestamps, speaker
labels; $0.22 per audio hour, cached per file), a local energy detector for speech regions without Scribe,
pyannote.audio's WeSpeaker ResNet34 (voiceprints, on MPS) and Praat through parselmouth (pitch).
"""
import hashlib, json, os, re, shutil, subprocess, unicodedata, urllib.request, warnings
from functools import lru_cache
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
SR = 16000
WIN = 6 * SR  # voiceprint chunk length
CACHE = Path(os.environ.get("VOICE_CHECK_CACHE", Path.home() / "Library/Caches/voice-check"))
AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".webm"}


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# Transcripts ---------------------------------------------------------------------------------------------

def _elevenlabs():
    exe = shutil.which("elevenlabs") or os.path.expanduser("~/.bun/bin/elevenlabs")
    if not os.path.exists(exe):
        raise SystemExit("elevenlabs CLI not found (install it globally with Bun and log in)")
    return exe


def transcribe(path, h=None, language="it"):
    """Scribe v2 words of a file: [{start, end, text, speaker}], cached by file hash."""
    h = h or sha1(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"scribe_{h}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    print(f"  transcribing {Path(path).name} with Scribe v2", flush=True)
    cmd = [_elevenlabs(), "speech-to-text", "convert", "--model-id", "scribe_v2", "--file", str(path),
           "--diarize", "true", "--timestamps-granularity", "word", "--tag-audio-events", "false", "--format", "json"]
    if language:
        cmd += ["--language-code", language]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"Scribe failed on {Path(path).name}: {r.stderr.strip()[:500]}")
    words = [{"start": w["start"], "end": w["end"], "text": w["text"], "speaker": w.get("speaker_id") or "speaker_0"}
             for w in json.loads(r.stdout)["words"] if w["type"] == "word"]
    cached.write_text(json.dumps(words, ensure_ascii=False))
    return words


def norm(t):
    """Comparable form of a word: lowercase, no accents or punctuation, doubled letters single (vabbè = vabè)."""
    t = unicodedata.normalize("NFKD", t.lower())
    return re.sub(r"(.)\1+", r"\1", re.sub(r"\W", "", "".join(c for c in t if not unicodedata.combining(c))))


# Numbers are compared as spoken words: 7 = sette, 2-3 = da due a tre, 18.44 = 18:44 = diciotto e quarantaquattro.
SPOKEN = {"it": {"from": "da", "to": "a", "past": "e", "point": "virgola", "percent": "per cento", "links": {"e", "a", "o"}}}
SPOKEN_DEFAULT = {"from": "", "to": "", "past": "", "point": "", "percent": "", "links": set()}


@lru_cache(maxsize=None)
def _number_words(language):
    """Every comparable word that spells a number from 0 to 1000 in the language."""
    from num2words import num2words
    return {norm(w) for n in range(1001) for w in num2words(n, lang=language).split()}


def _spoken(word, language):
    """Spoken words of one written word if it is a number, a range, a time or a decimal; else None."""
    from num2words import num2words
    say = lambda n: num2words(int(n), lang=language).split()
    ph = SPOKEN.get(language, SPOKEN_DEFAULT)
    core = re.sub(r"^[^\w]+|[^\w%°ºª]+$", "", word)
    pct = core.endswith("%")
    core = core.rstrip("%")
    if m := re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", core):               # 2-3: da due a tre
        out = [ph["from"], *say(m[1]), ph["to"], *say(m[2])]
    elif m := re.fullmatch(r"(\d{1,2})[.:](\d{2})", core):              # 18.44 / 18:44: diciotto e quarantaquattro
        h, mi = int(m[1]), int(m[2])
        out = [*say(h), ph["past"], *say(mi)] if mi else say(h)
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", core):                    # 3.500: tremilacinquecento
        out = say(core.replace(".", ""))
    elif m := re.fullmatch(r"(\d+),(\d+)", core):                       # 3,5: tre virgola cinque
        out = [*say(m[1]), ph["point"], *say(m[2])]
    elif m := re.fullmatch(r"(\d+)[°ºª]", core):                         # 1°: primo
        out = num2words(int(m[1]), lang=language, to="ordinal").split()
    elif core.isdigit():
        out = say(core)
    else:
        return None
    return out + (ph["percent"].split() if pct else [])


def tokens(words, language="it"):
    """Comparable tokens of a list of words, and for each token the index of the word it came from.

    Numbers become words, so a script with digits aligns with a transcript with spoken numbers and back.
    Links between two numbers ("due o tre", "diciotto e quarantaquattro") are dropped on both sides,
    so the same time or range read with or without them still matches."""
    toks, idx = [], []
    for i, w in enumerate(words):
        for t in _spoken(w, language) or [w]:
            if t := norm(t):
                toks.append(t); idx.append(i)
    links = {norm(x) for x in SPOKEN.get(language, SPOKEN_DEFAULT)["links"]}
    if links:
        nums = _number_words(language)
        keep = [not (t in links and 0 < k < len(toks) - 1 and toks[k - 1] in nums and toks[k + 1] in nums)
                for k, t in enumerate(toks)]
        toks, idx = [t for t, ok in zip(toks, keep) if ok], [i for i, ok in zip(idx, keep) if ok]
    return toks, idx


def cut_time(words, phrase):
    """End time of the first occurrence of phrase in the transcript, or None."""
    toks, _ = tokens(phrase.split())
    ws, idx = tokens([w["text"] for w in words])
    for i in range(len(ws) - len(toks) + 1):
        if toks and ws[i:i + len(toks)] == toks:
            return words[idx[i + len(toks) - 1]]["end"]
    return None


def speech_mask(words, n, pad=0.05, bridge=0.3):
    """Sample mask of speech from word timestamps; pauses shorter than `bridge` seconds count as speech."""
    mask = np.zeros(n, bool)
    prev = None
    for w in sorted(words, key=lambda w: w["start"]):
        a, b = w["start"] - pad, w["end"] + pad
        if prev is not None and a - prev < bridge:
            a = prev
        mask[max(0, int(a * SR)):min(n, int(b * SR))] = True
        prev = b
    return mask


def energy_mask(x, pad=0.05, bridge=0.3, shortest=0.12):
    """Sample mask of speech from loudness alone, free and offline; for recordings with one voice.

    A 10 ms frame is speech when it is 12 dB above the quietest tenth of the file and no more than
    35 dB below its loudest frames. Gaps shorter than `bridge` seconds count as speech, as in speech_mask."""
    hop = SR // 100
    n = len(x) // hop
    mask = np.zeros(len(x), bool)
    if not n:
        return mask
    rms = np.sqrt(np.convolve((x[:n * hop].reshape(n, hop) ** 2).mean(1), np.ones(3) / 3, "same"))
    db = 20 * np.log10(rms + 1e-9)
    on = db > max(np.percentile(db, 10) + 12, np.percentile(db, 99) - 35)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
    segs = []
    for a, b in zip(edges[0::2] / 100, edges[1::2] / 100):
        if segs and a - segs[-1][1] < bridge:
            segs[-1][1] = b
        else:
            segs.append([a, b])
    for a, b in segs:
        if b - a >= shortest:
            mask[max(0, int((a - pad) * SR)):min(len(x), int((b + pad) * SR))] = True
    return mask


# Other voices --------------------------------------------------------------------------------------------

def other_voices(language, gender, n=20):
    """Preview clips of up to n shared ElevenLabs voices in a language and gender, downloaded once.

    Listing the voice library and downloading previews are free (no credits). They are the "another voice"
    anchor of the fidelity score. Returns the clip paths; an empty list when the library can't be reached."""
    d = CACHE / "others" / f"{language}_{gender}"
    d.mkdir(parents=True, exist_ok=True)
    have = sorted(p for p in d.iterdir() if p.suffix == ".mp3")
    if len(have) >= n:
        return have[:n]
    print(f"  downloading {language} {gender} voice previews from the ElevenLabs voice library (free)", flush=True)
    r = subprocess.run([_elevenlabs(), "voices", "get_shared", "--language", language, "--gender", gender,
                        "--page-size", str(2 * n), "--format", "json", "--query", "voices[].[voice_id, preview_url]"],
                       capture_output=True, text=True)
    if r.returncode:
        print(f"  voice library unavailable: {r.stderr.strip()[:200]}", flush=True)
        return have
    for vid, url in json.loads(r.stdout or "[]"):
        if len(have) >= n:
            break
        out = d / f"{vid}.mp3"
        if out.exists() or not url:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "evaluate-voice-quality"})
            out.write_bytes(urllib.request.urlopen(req, timeout=30).read())
            have.append(out)
        except Exception:
            continue
    return sorted(have)[:n]


# Voiceprints ---------------------------------------------------------------------------------------------

class Voiceprint:
    """WeSpeaker ResNet34 speaker embeddings (pyannote.audio, ungated model), on MPS when available.
    The model loads on first use, so fully cached runs skip it."""

    model = None

    def _load(self):
        import torch
        from pyannote.audio import Model
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"  # MPS is ~3x faster on this Mac
        self.model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM").eval().to(self.dev)

    def __call__(self, chunks, batch=64):
        """Unit embeddings (N, 256) of equal-length chunks (N, samples)."""
        if self.model is None:
            self._load()
        out = []
        with self.torch.inference_mode():
            for b in range(0, len(chunks), batch):
                x = self.torch.from_numpy(np.ascontiguousarray(chunks[b:b + batch]))[:, None].to(self.dev)
                out.append(self.model(x).cpu().numpy())
        return unit(np.concatenate(out))

    def of(self, x, mask, win=WIN):
        """Embeddings of the masked speech in x, as consecutive `win`-sample chunks, with each chunk's start time."""
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
        a, b = edges[0::2], edges[1::2]
        if not len(a):
            return np.zeros((0, 256)), np.zeros(0)
        speech = np.concatenate([x[s:e] for s, e in zip(a, b)])
        cum = np.concatenate([[0], np.cumsum(b - a)])
        if len(speech) < win:  # too short for one full chunk: embed what there is
            return self(speech[None]), np.array([a[0] / SR])
        starts = list(range(0, len(speech) - win + 1, win))
        T = [(a[i] + s - cum[i]) / SR for s in starts for i in [np.searchsorted(cum, s, side="right") - 1]]
        return self(np.stack([speech[s:s + win] for s in starts])), np.array(T)


# Pitch ---------------------------------------------------------------------------------------------------

def f0_stats(x):
    """Median F0 (Hz), SD and 5-95% range (semitones) of a voiced signal, from Praat."""
    import parselmouth
    f = parselmouth.Sound(x, SR).to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=300).selected_array["frequency"]
    st = 12 * np.log2(f[f > 0] / 100)
    return np.array([np.median(f[f > 0]), np.std(st), np.percentile(st, 95) - np.percentile(st, 5)])
