# evaluate-voice-quality

Controlla una traccia vocale generata dall'IA rispetto al copione e dà a ogni take di una voce clonata un punteggio da 0 a 100 di fedeltà alla voce reale.

Fa parte della [Prompt Library di CH Labs](https://www.cryptohelvetia.com/ch-labs/prompt-library), dove la pagina di riferimento è [Evaluate voice quality](https://www.cryptohelvetia.com/ch-labs/prompt-library/skills/evaluate-voice-quality). Il repo contiene una copia della skill, rigenerata a ogni aggiornamento della libreria.

## Installare

In Claude Code aggiungi una volta il marketplace e installa il plugin.

```
/plugin marketplace add MicheDada21/ch-labs-skills
/plugin install evaluate-voice-quality@ch-labs-skills
```

Da shell i comandi sono `claude plugin marketplace add MicheDada21/ch-labs-skills` e `claude plugin install evaluate-voice-quality@ch-labs-skills`. Nell'app desktop premi + accanto al campo del prompt, apri Plugins, aggiungi il marketplace `MicheDada21/ch-labs-skills` con Add marketplace e poi il plugin con Add plugin.

Per avere tutte le skill della libreria installa `tutte-le-skill@ch-labs-skills`. Con altri agenti copia la cartella `skills/evaluate-voice-quality` tra le skill dell'agente, per esempio in `~/.codex/skills/evaluate-voice-quality`.

## Requisiti

Servono [uv](https://docs.astral.sh/uv/), ffmpeg e la CLI `elevenlabs` installata con Bun e collegata al tuo account. Al primo avvio uv installa l'ambiente Python, Torch compreso, in qualche minuto.

## Licenza

MIT, testo completo in [LICENSE](LICENSE).
