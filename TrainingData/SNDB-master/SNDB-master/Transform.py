import mido #type: ignore
from midi2audio import FluidSynth #type: ignore
from glob import glob
from os.path import join

def split_midi(midi_file, segment_length, output_folder):
    mid = mido.MidiFile(midi_file)
    ticks_per_beat = mid.ticks_per_beat
    segment_ticks = int(segment_length * mid.ticks_per_beat)

    for dump, track in enumerate(mid.tracks):
        start_tick = 0
        segment_num = 0
        while start_tick < track.length:
            new_track = mido.MidiTrack()
            new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
            new_midi.tracks.append(new_track)

            for msg in track:
                if msg.time >= start_tick and msg.time < start_tick + segment_ticks:
                    new_track.append(msg)
            
            output_midi = f"{output_folder}/segment_{segment_num}.mid"
            new_midi.save(output_midi)

            output_mp3 = f"{output_folder}/segment_{segment_num}.mp3"
            FluidSynth().midi_to_audio(output_midi, output_mp3)
            
            segment_num += 1
            start_tick += segment_ticks


path = 'C:\\Users\\studi\\TrainingData\\SNDB-master\\SNDB-master\\build\\VSL'
mid_files = glob(join(path, '*.mid'))

for mid_file in mid_files:

    print(f"Processing {mid_file}")
    split_midi(mid_file, 6, 'C:\\Users\\studi\\TrainingData\\SNDB-file')
