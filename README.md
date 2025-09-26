# musicxml2midi – Piano Roll Inspector

Konvertiert **MusicXML** in MIDI-nahe Datenstrukturen und zeigt sie in einer **GUI (PySide6 + PyQtGraph)** an.

## Features (bisher)

- **MusicXML-Parsing**
  - Noten (Pitch, Start/Ende in Ticks, Velocity)
  - Dynamiken (`pp..ff`), Hairpins (crescendo/diminuendo)
  - Artikulationen (z. B. `slur`, `staccato`, `tenuto`, `accent`, `marcato`, `pizzicato`, `tremolo`, `mute`)
  - Tempo- und Taktartwechsel

- **GUI**
  - Piano Roll (Noten farbcodiert nach Velocity: blau → rot)
  - Dynamics-Lane (Labels + farbiges, halbtransparentes Band für Hairpins)
  - Articulations-Lane (mehrere gleichzeitige Artikulationen, gleichmäßig vertikal verteilt)
  - Beat- und Taktlinien mit Zählung

- **Interpretation**
  - Slur-Erkennung onset-basiert pro Voice/Staff (Legato-Übergänge)
  - Synchronisierte Darstellung von Dynamik-/Artikulations-Events

## Geplante Erweiterungen

- **Vorher/Nachher-Panel**
  - Oben: Original (aus MusicXML)
  - Unten: interpretierte MIDI-Daten (Velocity-Lane, CC1/CC11-Lanes)

- **Bedienelemente**
  - „Settings“ (Humanisierung, Regeln)
  - „Apply to MIDI“ (Transformation)
  - „Export MIDI“

## Projektstruktur

musicxml2midi/
├─ src/musicxml2midi/
│  ├─ gui/
│  │  ├─ app.py            # Main GUI, Datei laden, Panels
│  │  ├─ pianoroll.py      # Piano Roll (Notes/Dynamics/Arts)
│  │  ├─ pianoroll_post.py # (geplant) Nachher-Panel
│  │  └─ utils.py          # GestureViewBox etc.
│  │  └─ settings.py       # Settings-Dialog
│  │  └─ models.py         # Datenklassen für GUI (MidiSong, Note, ...)
│  ├─ humanize/          # (geplant) Humanisierungs-Regeln
│  │  └─ velocity.py         # Regeln (Velocity, Timing, ...)
│  ├─ config.py         # Config laden (z. B. Default-Pfade)
│  ├─ analyze.py        # MusicXML → interne Tokens/Events
│  ├─ process.py        # Events → Timeline/MidiSong
│  └─ timeline.py       # Datenklassen
└─ examples/
   └─ mar.musicxml      # Beispiel (optional Auto-Load)


## Installation & Start
pip install -r requirements.txt
python gui/app.py


## Todos:
- [x] 1. Settings sollen einen Reset-Button und einen Export und Import-Settings Dialog.
- [x] 2. Velocities sollen Humanisierung neben Beat-Akzentuierung bekommen.
- [x] 3. Noten Positionen sollen Humanisierung bekommen.
  - [x] Noten-Timing sollte CC1/CC11-Timing beeinflussen
- [x] 4. Noten sollen sich bei Slurs überlappen (unabhängig von der Humanisierung der Notenpositionen).
	- [x] a) Setting welche beeinflusst, wie viele Ticks sich die Noten überlappen.
	- [x] b) Setting welche beeinflusst, um wie viele Ticks die geslurte Note (also jede Note in einer Phrase, die nicht die erste ist) früher startet - sowohl absolut als auch relativ zur Notenlänge. Weißt du wie ich meine? Sodass längere geslurte Noten schon früher starten, als kürzere Noten).
	- [ ] c) Setting welche einstellen lässt, dass die Velocity nur bei der ersten Note in einer Phrase von der Dynamik abhängt, die anderen Noten (also die geslurten Noten) sollen von der Länge der Note abhängen, also leichtere Anschlagstärke bei längeren Noten - ergibt das Sinn?
- [ ] 5. Midi-Export
- [ ] 6. UI Speed Optimierung