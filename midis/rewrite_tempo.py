import pretty_midi
import os
import sys

pm = pretty_midi.PrettyMIDI(sys.argv[1])
# x2 multiplier to tempo in the html
change_bpm = 160.
pm._tick_scales.append(
    (pm.time_to_tick(0.), 60./(change_bpm*pm.resolution)))
pm._update_tick_to_time(pm.time_to_tick(pm.get_end_time()))
pm.write("out.mid")
