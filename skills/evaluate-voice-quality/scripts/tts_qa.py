"""Check a generated voice track against its script: words, voices, drift, pace and pitch.

Usage (S = this scripts folder):
  uv run --project "$S" "$S/tts_qa.py" take.mp3 script.txt
  uv run --project "$S" "$S/tts_qa.py" take.mp3 script.txt --voice "TB=real voice.mp3" --voice "Luca=luca.mp3"

Script: plain text. Lines written as "Name: text" belong to that character; other lines to the narrator,
named by --narrator. Audio tags in square brackets, like [whispers], are ignored.
--voice NAME=FILE enrolls a recording of a character's real or intended voice: lines are then checked against it.
Without it, each character's lines are checked against that character's own average voice.
The audio is transcribed once with ElevenLabs Scribe v2 ($0.22 per audio hour, cached in ~/Library/Caches/voice-check).
"""
import argparse, difflib, re
from pathlib import Path
import numpy as np
from voicekit import SR, Voiceprint, decode, f0_stats, norm, sha1, speech_mask, tokens, transcribe, unit

TAG = re.compile(r"\[[^\]]*\]")
SPEAKER = re.compile(r"^\s*([A-ZÀ-Ý][^\W\d_]*(?: [A-ZÀ-Ý][^\W\d_]*){0,2})\s*:\s+(.*)$")  # "Name: text", up to 3 capitalized words


def parse_script(text, narrator):
    """[(character, line text)] from a script, one entry per non-empty line."""
    out = []
    for raw in text.splitlines():
        raw = TAG.sub(" ", raw)
        m = SPEAKER.match(raw)
        who, line = (m.group(1), m.group(2)) if m else (narrator, raw)
        line = line.strip()
        if any(norm(t) for t in line.split()):
            out.append((who, line))
    return out


def clock(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="generated voice track")
    ap.add_argument("script", type=Path, help="text the track should read (UTF-8)")
    ap.add_argument("--voice", action="append", default=[], metavar="NAME=FILE",
                    help="recording of a character's voice (repeatable)")
    ap.add_argument("--narrator", default="Narrator", help="character name for lines without a 'Name:' prefix")
    ap.add_argument("--language", default="it", help="ISO 639-1 language of the track (default: it)")
    ap.add_argument("--report", type=Path, help="markdown report path (default: <audio>.qa.md)")
    a = ap.parse_args()
    report = a.report or a.audio.with_suffix(".qa.md")
    vp = Voiceprint()

    # Words: align the script with what Scribe heard ---------------------------------------------------
    lines = parse_script(a.script.read_text(encoding="utf-8"), a.narrator)
    # Tokens, not words: numbers are spelled out (7 = sette, 2-3 = da due a tre), so one word can give several.
    sraw = [t for _, line in lines for t in line.split()]
    wline = [i for i, (_, line) in enumerate(lines) for _ in line.split()]
    stoks, sidx = tokens(sraw, a.language)
    sline = [wline[k] for k in sidx]
    heard = transcribe(a.audio, sha1(a.audio), a.language)
    htoks, hidx = tokens([w["text"] for w in heard], a.language)
    sm = difflib.SequenceMatcher(None, stoks, htoks, autojunk=False)
    span = [[np.inf, -np.inf] for _ in lines]
    heard_of_line = [set() for _ in lines]
    issues = []
    uniq = lambda xs: list(dict.fromkeys(xs))
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        for k in range(min(i2 - i1, j2 - j1)) if op in ("equal", "replace") else []:
            li, w = sline[i1 + k], heard[hidx[j1 + k]]
            span[li] = [min(span[li][0], w["start"]), max(span[li][1], w["end"])]
            heard_of_line[li].add(hidx[j1 + k])
        if op == "equal":
            continue
        at = heard[hidx[j1]]["start"] if j1 < len(hidx) else (heard[-1]["end"] if heard else 0)
        said = " ".join(heard[j]["text"] for j in uniq(hidx[j1:j2]))
        meant = " ".join(sraw[k] for k in uniq(sidx[i1:i2]))
        kind = {"replace": "misread", "delete": "skipped", "insert": "extra"}[op]
        issues.append((at, kind, meant, said, sline[i1] if i1 < len(sline) else sline[-1]))
    words_of_line = [[heard[j] for j in sorted(s)] for s in heard_of_line]
    errors = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    wer = errors / max(1, len(stoks))

    # Voices: embed each line and each character ------------------------------------------------------
    x = decode(a.audio)
    chars = sorted({c for c, _ in lines})
    line_emb = {}
    for i, ws in enumerate(words_of_line):
        if ws and span[i][1] - span[i][0] >= 1.0:
            E, _ = vp.of(x, speech_mask(ws, len(x)))
            line_emb[i] = unit(E.mean(0))
    enrolled = {}
    for spec in a.voice:
        name, _, path = spec.partition("=")
        p = Path(path)
        v = decode(p)
        E, _ = vp.of(v, speech_mask(transcribe(p, sha1(p), a.language), len(v)))
        enrolled[name.strip()] = unit(E.mean(0))
    ref = dict(enrolled)
    for c in chars:  # characters without a recording are compared with their own average
        own = [line_emb[i] for i in line_emb if lines[i][0] == c]
        if c not in ref and own:
            ref[c] = unit(np.mean(own, 0))
    names = sorted(ref)
    wrong = []
    if len(names) > 1:
        R = np.stack([ref[n] for n in names])
        for i, e in line_emb.items():
            sims = R @ e
            got = names[int(sims.argmax())]
            if lines[i][0] in ref and got != lines[i][0] and sims.max() - sims[names.index(lines[i][0])] >= 0.1:
                wrong.append((span[i][0], lines[i][0], got, sims.max() - sims[names.index(lines[i][0])], lines[i][1]))

    # Drift, pace and pitch per character ---------------------------------------------------------------
    rows, drift = [], []
    for c in chars:
        ws = [w for i, wl in enumerate(words_of_line) if lines[i][0] == c for w in wl]
        if not ws:
            rows.append(f"| {c} | 0 | – | – | – | – | – |"); continue
        m = speech_mask(ws, len(x))
        talk = m.sum() / SR
        wpm = 60 * len(ws) / talk
        f = f0_stats(x[m]) if talk > 3 else np.full(3, np.nan)
        E, T = vp.of(x, m)
        sim = E @ ref[c] if c in ref else np.full(len(E), np.nan)
        rows.append(f"| {c} | {len(ws)} | {talk:.0f} s | {wpm:.0f} | {f[0]:.0f} Hz | {f[2]:.1f} st | "
                    f"{np.nanmedian(sim):.2f} |")
        if len(E) >= 6:  # chunks far below the character's usual similarity
            med = np.median(sim); mad = 1.4826 * np.median(np.abs(sim - med)) + 1e-3
            drift += [(T[k], c, sim[k], (sim[k] - med) / mad) for k in range(len(E)) if sim[k] < med - 4 * mad]
    pace = []
    rates = {i: len(words_of_line[i]) / (span[i][1] - span[i][0]) * 60 for i in line_emb}
    if len(rates) >= 4:
        med = np.median(list(rates.values()))
        pace = [(span[i][0], lines[i][0], r, lines[i][1]) for i, r in rates.items() if abs(r / med - 1) > 0.35]
    gaps = [(heard[k]["end"], heard[k + 1]["start"] - heard[k]["end"]) for k in range(len(heard) - 1)
            if heard[k + 1]["start"] - heard[k]["end"] > 2.0]

    # Report ----------------------------------------------------------------------------------------------
    cut = lambda s, n=70: s if len(s) <= n else s[:n - 1] + "…"
    L = [f"# TTS QA: {a.audio.name}", "", f"Script: `{a.script.name}` · {len(lines)} lines, {len(set(sidx))} words · "
         f"heard {len(heard)} words · word error rate {wer:.1%}", "",
         f"Voices checked against: " + (", ".join(f"{n} (recording)" if n in enrolled else f"{n} (own average)"
                                               for n in names) or "–"), "",
         "## Voices", "", "| Character | Words | Speech | Words/min | Median F0 | F0 range | Similarity to voice |",
         "|---|---|---|---|---|---|---|", *rows, "",
         f"## Words not read as written ({len(issues)})", ""]
    L += (["| Time | Kind | Script | Heard | Line by |", "|---|---|---|---|---|"]
          + [f"| {clock(t)} | {k} | {cut(m, 40)} | {cut(s, 40)} | {lines[li][0]} |" for t, k, m, s, li in issues]
          if issues else ["None."])
    L += ["", f"## Lines in the wrong voice ({len(wrong)})", ""]
    L += (["| Time | Should be | Sounds like | Margin | Line |", "|---|---|---|---|---|"]
          + [f"| {clock(t)} | {w} | {g} | {d:.2f} | {cut(s)} |" for t, w, g, d, s in sorted(wrong)]
          if wrong else ["None."])
    L += ["", f"## Voice drift ({len(drift)} chunks of 6 s)", ""]
    L += (["| Time | Character | Similarity | Robust z |", "|---|---|---|---|"]
          + [f"| {clock(t)} | {c} | {s:.2f} | {z:.1f} |" for t, c, s, z in sorted(drift)] if drift else ["None."])
    L += ["", f"## Pace outliers ({len(pace)} lines more than 35% off the median rate)", ""]
    L += (["| Time | Character | Words/min | Line |", "|---|---|---|---|"]
          + [f"| {clock(t)} | {c} | {r:.0f} | {cut(s)} |" for t, c, r, s in sorted(pace)] if pace else ["None."])
    L += ["", f"## Pauses over 2 s ({len(gaps)})", ""]
    L += [", ".join(f"{clock(t)} ({g:.1f} s)" for t, g in gaps) if gaps else "None."]
    report.write_text("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
