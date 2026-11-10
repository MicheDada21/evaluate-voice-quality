---
name: evaluate-voice-quality
description: "Flags misread words, wrong-voice lines and drift in AI speech, and scores each voice-clone take 0-100 against the real speaker. Use when checking a generated voice track or ranking clone takes."
---

# Evaluate voice quality

Two scripts share one core (`scripts/voicekit.py`). `tts_qa.py` checks one generated track against the text it should read. `voice_match.py` gives one take or a folder of takes a 0-100 fidelity to a recording of the real voice and ranks them. Both write a Markdown report and leave the audio untouched.

Words come from ElevenLabs Scribe v2 (word timestamps and speaker labels, $0.22 per audio hour). `voice_match.py --free` skips Scribe and finds speech offline from loudness, so it costs nothing. Voiceprints come from pyannote.audio's WeSpeaker ResNet34 on MPS, pitch from Praat. Numbers count as read correctly whether written as digits or spoken (7 and sette, 2-3 and da due a tre, 18.44 and diciotto e quarantaquattro). Word comparison ignores accents, punctuation and doubled letters.

## Steps

1. Pick the script. Use `tts_qa.py` for a single track that has a text, to learn whether it read the script in the right voices. Use `voice_match.py` for one take or a folder of takes plus a long recording of the real speaker, to learn how faithful each is (0-100) and which sounds most like them. Add `--free` when every take has a single voice and the user wants no cost. Run both when the user wants the best take and a check of it. Done when you know which files go to which script.

2. Prepare the script text for `tts_qa.py`. Plain UTF-8 text. A line written as `Name: text` belongs to that character; every other line belongs to the narrator, named with `--narrator`. Audio tags in square brackets, like `[whispers]`, are ignored. When the character of each dialogue line is a guess, tell the user which lines you assigned. Done when the text file exists with speaker names.

3. Run the script from any folder. The user supplying the audio counts as approval for the Scribe charge; each file is transcribed once and cached. `--free` runs and the other-voice previews cost nothing.

       S=<this skill's folder>/scripts
       uv run --project "$S" "$S/tts_qa.py" take.mp3 script.txt --narrator TB [options]
       uv run --project "$S" "$S/voice_match.py" "real voice.mp3" takes/ [options]
       uv run --project "$S" "$S/voice_match.py" "real voice.mp3" take.mp3 --free

   | Option | Script | When |
   |---|---|---|
   | `--voice NAME=FILE` | tts_qa | A recording of a character's real or intended voice exists; lines are checked against it. Repeat per character. Without it each character is checked against its own average. |
   | `--language it` | tts_qa | Track language, ISO 639-1 (default it). Number spelling follows it. |
   | `--free` | voice_match | Find speech offline instead of with Scribe. No cost; no second-voice removal, so use it on single-voice takes or with `--cut-at`. |
   | `--cut-after "phrase"` | voice_match | Score only up to the end of a phrase, for example to stop before a dialogue with another voice. Needs Scribe. |
   | `--cut-at SECONDS` | voice_match | Score only the first SECONDS of each take. Works with `--free`. |
   | `--others DIR` | voice_match | Recordings of other speakers to use as the fidelity floor instead of the ElevenLabs voice library previews. |
   | `--language it` | voice_match | Language of the other voices in the fidelity floor (default it). |
   | `--exclude TEXT` | voice_match | Skip takes whose file name contains TEXT. Repeatable. |
   | `--report file.md` | both | Report path. Defaults to `<audio>.qa.md`, `<takes>/voice-match-report.md`, or `<take>.voice-match.md` for one take. |

   Done when the script exits 0 and prints the report path.

4. Read the report and answer. For `tts_qa.py`, list the words not read as written, lines in the wrong voice and drift chunks with their times (m:ss), so the user can jump to them. For `voice_match.py`, give each take's fidelity with its 95% range, the top takes and whether there is a confident winner; when the report says the top takes are tied, suggest more takes per setting or a choice by ear. Done when the answer names times or files the user can act on.

## Notes and limits

- First run installs the Python environment with uv (Torch included, a few minutes); later runs start in seconds. The model downloads once from Hugging Face without a token.
- Needs ffmpeg and the `elevenlabs` CLI, installed globally with Bun and logged in. A missing CLI stops the run with a message.
- Cache: `~/Library/Caches/voice-check`, keyed by file hash (override with `VOICE_CHECK_CACHE`). Transcripts and embeddings are reused, so reruns are free. Rename a take freely; editing its audio makes a new transcript.
- Scribe labels a second speaker in dialogue takes. `voice_match.py` removes that voice before scoring, so only the cloned voice is ranked.
- Fidelity: 100 means as close to the real voice as the speaker is to themself across passages of the same length; 0 means as close as the typical other voice of the same language and gender. The floor comes from up to 20 free preview clips of the ElevenLabs voice library, matched to the reference's pitch (male below 165 Hz), downloaded once to the cache. Without the library or `--others`, the report gives only the Score.
- Fidelity is a speaker-ID measure: it tells how much a take sounds like the same person to a voiceprint model, not how natural or well read it is. Compare takes of the same text and length; previews are short, so the floor sits a little low and fidelity a little high. Overlapping 95% ranges mean the takes are close to a tie.
- Score is the same distance in units of the real voice's own variation (0 means as close as the speaker is to themself).
- A foreign word can come back spelled phonetically ("dehors" heard as "The Horse"). Listen to the spot before regenerating.
- Voiceprints run on MPS when available, otherwise on CPU (about three times slower).
