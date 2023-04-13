# SUNMASK

SUNMASK was presented at EvoMUSART 2023, slides can be viewed here [https://docs.google.com/presentation/d/1ne2jtwl3xVq9Vy3psoUF3qU3VKYYpBp3F-Mv_u_Xd5E/edit?usp=sharing](https://docs.google.com/presentation/d/1ne2jtwl3xVq9Vy3psoUF3qU3VKYYpBp3F-Mv_u_Xd5E/edit?usp=sharing)

Code, see `code/README` for details on the various released code formats.



To try the MIDI sample player as a demo, see [https://sunmask-web.github.io/SUNMASK/](https://sunmask-web.github.io/SUNMASK/)

There is a known outstanding bug with MIDIjs, regarding the player skipping the first note, and also skipping some note playback late in the file. We will try to correct this or switch to a different player in the future.

Additionally, many of the midi files for coconet are not well processed by midijs, we are working on a fix but this is why many coconet samples do not show up in the web player. `coconet_sample_1_0.mid` is a generally well functioning example, which shows the quality of Coconet as a generative model.



The original samples page can also be seen at [https://coconets.github.io/](https://coconets.github.io/), or interactively via the Google Doodle [https://www.google.com/doodles/celebrating-johann-sebastian-bach](https://www.google.com/doodles/celebrating-johann-sebastian-bach) or glitch.me versions [http://coconet.glitch.me/](http://coconet.glitch.me/).

As an alternative, you can use the timidity program `timidity -T 160 file.mid` on any of the files from `midis/` , downloaded to your local machine.
