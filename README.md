# SUNMASK

To try the MIDI sample player, see [https://sunmask-web.github.io/SUNMASK/](https://sunmask-web.github.io/SUNMASK/)



There is a known outstanding bug with MIDIjs, regarding the player skipping the first note, and also skipping some note playback late in the file. We will try to correct this or switch to a different player in the future.

As an alternative, you can use `timidity -T 160 file.mid` on any of the files from midis/ , downloaded to your local browser.
