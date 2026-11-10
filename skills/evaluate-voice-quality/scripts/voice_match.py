"""Score generated takes for how faithful they are to a reference recording of the real voice, and rank them.

Usage (S = this scripts folder):
  uv run --project "$S" "$S/voice_match.py" "real voice.mp3" takes/
  uv run --project "$S" "$S/voice_match.py" "real voice.mp3" take.mp3 --free
  uv run --project "$S" "$S/voice_match.py" "real voice.mp3" takes/ --exclude _m2
  uv run --project "$S" "$S/voice_match.py" "real voice.mp3" takes/ --cut-after "stretta di mano"

Fidelity (0-100): 100 = as close as the real voice is to itself, 0 = as close as other voices of the same
language and gender (free preview clips from the ElevenLabs voice library, or --others).

By default every file is transcribed once with ElevenLabs Scribe v2 ($0.22 per audio hour, cached in
~/Library/Caches/voice-check), for speech regions and to remove a second voice. --free finds speech regions
offline from loudness instead: no cost, for takes with a single voice.
"""
import argparse, hashlib, sys
from pathlib import Path
import numpy as np
from voicekit import (AUDIO, CACHE, SR, Voiceprint, cut_time, decode, energy_mask, f0_stats, other_voices, sha1,
                      speech_mask, transcribe, unit)

SAME_SPEAKER = 0.5  # WeSpeaker cosine between speaker centroids; below it a Scribe speaker is another voice
MALE_F0 = 165  # Hz; a reference with a lower median pitch is compared with male voices


def analyze(path, h, key, vp, is_ref, free, ref_centroid=None, cut=None):
    cached = CACHE / f"{key}.npz"
    if cached.exists():
        return dict(np.load(cached))
    words = None if free else transcribe(path, h)
    print(f"  embedding {path.name}", flush=True)
    x = decode(path)
    if cut is not None:
        x = x[:int((cut + 0.15) * SR)]
        words = words and [w for w in words if w["start"] < cut + 0.15]
    sp = energy_mask(x) if free else speech_mask(words, len(x))
    keep, removed = sp, 0.0
    speakers = sorted({w["speaker"] for w in words or []})
    if ref_centroid is not None and len(speakers) > 1:  # drop a dialogue counterpart labeled by Scribe
        cen = {s: unit(vp.of(x, speech_mask([w for w in words if w["speaker"] == s], len(x)))[0].mean(0))
               for s in speakers}
        narrator = max(speakers, key=lambda s: cen[s] @ ref_centroid)
        other = [s for s in speakers if cen[s] @ cen[narrator] < SAME_SPEAKER]
        if other:
            keep = sp & ~speech_mask([w for w in words if w["speaker"] in other], len(x))
            removed = (sp & ~keep).sum() / SR
    E, T = vp.of(x, keep)
    d = {"wespeaker": E, "t": T, "removed": np.array(removed), "cut": np.array(-1.0 if cut is None else cut)}
    n = 120 * SR
    d["f0"] = (np.stack([f0_stats(x[i:i + n]) for i in range(0, len(x) - n, max(n, len(x) // 40))])
               if is_ref else f0_stats(x[keep])[None])
    np.savez(cached, **d)
    return d


def other_centroids(paths, vp):
    """Voiceprint centroid of each other-voice clip (offline speech regions, cached by file hash)."""
    out = []
    for p in paths:
        cached = CACHE / f"other_{sha1(p)}.npz"
        if not cached.exists():
            x = decode(p)
            E, _ = vp.of(x, energy_mask(x))
            np.savez(cached, wespeaker=E)
        E = np.load(cached)["wespeaker"]
        if len(E):
            out.append(unit(E.mean(0)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path, help="long recording of the real voice")
    ap.add_argument("experiments", type=Path, help="a generated take, or a folder of takes")
    ap.add_argument("--free", action="store_true",
                    help="find speech offline from loudness instead of Scribe: no cost, one voice per take")
    ap.add_argument("--exclude", action="append", default=[], metavar="TEXT",
                    help="skip takes whose file name contains TEXT (repeatable), e.g. --exclude _m2")
    ap.add_argument("--cut-after", metavar="PHRASE",
                    help="score only up to the end of PHRASE (first occurrence in the transcript), "
                         'e.g. --cut-after "stretta di mano" to stop before a dialogue; needs Scribe')
    ap.add_argument("--cut-at", type=float, metavar="SECONDS", help="score only the first SECONDS of each take")
    ap.add_argument("--others", type=Path, metavar="DIR",
                    help="folder of recordings of other speakers for the fidelity floor "
                         "(default: ElevenLabs voice library previews in the reference's language and gender)")
    ap.add_argument("--language", default="it", help="ISO 639-1 language of the other voices (default it)")
    ap.add_argument("--report", type=Path, help="markdown report path (default: <experiments>/voice-match-report.md)")
    a = ap.parse_args()
    if a.free and a.cut_after:
        sys.exit("--cut-after needs a Scribe transcript; with --free use --cut-at SECONDS")
    CACHE.mkdir(parents=True, exist_ok=True)
    folder = a.experiments if a.experiments.is_dir() else a.experiments.parent
    report = a.report or (a.experiments.with_suffix(".voice-match.md") if a.experiments.is_file()
                          else folder / "voice-match-report.md")
    mode = "v" if a.free else "3"  # cache key: offline speech regions or Scribe

    vp = Voiceprint()
    rh = sha1(a.reference)
    ref = analyze(a.reference, rh, f"ref{mode}_{rh}", vp, True, a.free)
    ref_centroid = unit(ref["wespeaker"].mean(0))

    takes, dupes, seen, missing = {}, [], {}, []
    files = [a.experiments] if a.experiments.is_file() else sorted(a.experiments.iterdir())
    for p in files:
        if p.suffix.lower() not in AUDIO or any(e in p.name for e in a.exclude) or p.resolve() == a.reference.resolve():
            continue
        h = sha1(p)
        if h in seen:
            dupes.append((p.name, seen[h])); continue
        seen[h] = p.name
        cut, tag = a.cut_at, f"_c{a.cut_at:g}" if a.cut_at else ""
        if a.cut_after:
            cut = cut_time(transcribe(p, h), a.cut_after)
            if cut is None:
                missing.append(p.name); continue
            tag = "_" + hashlib.sha1(a.cut_after.encode()).hexdigest()[:8]
        takes[p.name] = analyze(p, h, f"take{mode}_{h}{tag}", vp, False, a.free, ref_centroid, cut)
    if len(takes) < 1:
        sys.exit("no audio files found in " + str(a.experiments))
    names = list(takes)
    block = max(5, int(np.median([len(takes[n]["wespeaker"]) for n in names])))  # real-voice blocks as long as a take

    rng = np.random.default_rng(0)
    X, t = ref["wespeaker"], ref["t"]
    s = X @ unit(X.mean(0))
    med, mad = np.median(s), np.median(np.abs(s - np.median(s)))
    inl = s > med - 4 * 1.4826 * mad  # drop chunks that are not the reference speaker (noise, other voices)
    Xi, ti = X[inl], t[inl]
    cen = unit(Xi.mean(0))
    gen = []
    for b in range(0, len(Xi) - block + 1, block):  # held-out blocks of the real voice, same length as a take
        far = (ti < ti[b] - 600) | (ti > ti[b + block - 1] + 600)
        gen.append(unit(Xi[b:b + block].mean(0)) @ unit(Xi[far].mean(0)))
    gen = np.array(gen); mu, sd = gen.mean(), gen.std()

    # Fidelity floor: how close other speakers of the same language and gender come to the real voice.
    gender = "male" if np.median(ref["f0"][:, 0]) < MALE_F0 else "female"
    if a.others:
        others = sorted(p for p in a.others.iterdir() if p.suffix.lower() in AUDIO)
        source = f"`{a.others.name}/`"
    else:
        others = other_voices(a.language, gender)
        source = f"ElevenLabs voice library previews ({a.language}, {gender})"
    oc = other_centroids(others, vp) if others else np.zeros((0, 256))
    floor = float(np.median(oc @ cen)) if len(oc) >= 3 else None
    fid = lambda c: np.clip(100 * (c - floor) / (mu - floor), 0, 100) if floor is not None else None

    cos, z, boot = {}, {}, {}
    for n in names:
        Y = takes[n]["wespeaker"]
        cos[n] = unit(Y.mean(0)) @ cen
        z[n] = (cos[n] - mu) / sd
        idx = rng.integers(0, len(Y), (2000, len(Y)))
        boot[n] = unit(Y[idx].mean(1)) @ cen

    cb = np.stack([boot[n] for n in names])
    pbest = dict(zip(names, np.bincount(cb.argmax(0), minlength=len(names)) / cb.shape[1]))
    order = sorted(names, key=lambda n: -z[n])
    rf = ref["f0"]; fmu, fsd = rf.mean(0), rf.std(0)

    best = order[0]
    fbest = f" Fidelity {fid(cos[best]):.0f}/100." if floor is not None else ""
    if len(names) == 1:
        verdict = (f"**Fidelity of `{best}`: {fid(cos[best]):.0f}/100.**" if floor is not None
                   else f"**`{best}`: {z[best]:+.1f} standard deviations from the real voice.**")
    elif pbest[best] >= 0.95:
        verdict = f"**Closest take: `{best}`** (best in {pbest[best]:.0%} of resamples).{fbest}"
    else:
        tied = [n for n in order if pbest[n] >= 0.05]
        verdict = ("**No confident single winner.** Tied for closest: " + ", ".join(f"`{n}` ({pbest[n]:.0%})" for n in tied)
                   + ". Generate more takes per setting or pick by ear.")

    regions = ("speech regions found offline from loudness (`--free`, no second-voice removal)" if a.free else
               "speech regions and speaker labels from ElevenLabs Scribe v2; a second voice (dialogue counterpart) "
               "labeled by Scribe is removed from each take, so only the cloned voice is scored")
    head = ["# Voice match report", "", f"Reference: `{a.reference.name}` · takes: {len(names)} unique", "", verdict, "",
            ("Fidelity = 0-100: 100 is as close as the real voice is to itself across passages of the same length, "
             f"0 is as close as the typical other voice ({len(oc)} from {source}). "
             f"95% of the real voice's own passages score {fid(np.percentile(gen, 5)):.0f} or more. " if floor is not None else
             "Fidelity unavailable: fewer than 3 other-voice recordings (pass `--others DIR`). ")
            + "Score = the same distance in standard deviations of the real voice's own passages (0 = as close as "
            "you are to yourself; lower = further). Speaker-ID model: WeSpeaker ResNet34 (pyannote.audio); "
            f"{regions}. Brackets = 95% range across chunk resamples; pitch is measured on the scored speech."
            + (f" Each take is cut after \"{a.cut_after}\" (cut time in the last column)." if a.cut_after else "")
            + (f" Each take is cut at {a.cut_at:g} s." if a.cut_at else ""), ""]
    table = ["| # | Take | Fidelity | 95% range | Score | P(best) | F0 Hz | F0 range st | 2nd voice removed | Cut at |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, n in enumerate(order, 1):
        f = takes[n]["f0"][0]
        if floor is not None:
            lo, hi = np.percentile(fid(boot[n]), [2.5, 97.5])
            fcol = f"{fid(cos[n]):.0f} | {lo:.0f} to {hi:.0f}"
        else:
            fcol = "– | –"
        table.append(f"| {i} | `{n}` | {fcol} | {z[n]:+.1f} | {pbest[n]:.0%} | {f[0]:.0f} | {f[2]:.1f} | "
                     f"{float(takes[n]['removed']):.0f} s | "
                     + (f"{float(takes[n]['cut']):.1f} s |" if float(takes[n]['cut']) >= 0 else "– |"))
    tail = ["", f"Real voice pitch: median F0 {fmu[0]:.0f} ± {fsd[0]:.0f} Hz, range {fmu[2]:.1f} ± {fsd[2]:.1f} semitones "
            "(per 2-minute block).", "", "## Anchors", "",
            "| Real voice vs itself (100) | SD | Worst real passage | Typical other voice (0) | Closest other voice | Other voices | Chunks dropped as not-you |",
            "|---|---|---|---|---|---|---|",
            f"| {mu:.3f} | {sd:.3f} | {gen.min():.3f} | " + (f"{floor:.3f} | {(oc @ cen).max():.3f} | {len(oc)} | "
                                                          if floor is not None else "– | – | 0 | ")
            + f"{(~inl).sum()}/{len(s)} |"]
    if missing:
        tail += ["", f"Skipped, phrase \"{a.cut_after}\" not found in transcript: " + ", ".join(f"`{m}`" for m in missing)]
    if dupes:
        tail += ["", "Skipped byte-identical duplicates: " + ", ".join(f"`{d}` (= `{o}`)" for d, o in dupes)]
    report.write_text("\n".join(head + table + tail) + "\n")
    print("\n".join([verdict, ""] + table))
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
