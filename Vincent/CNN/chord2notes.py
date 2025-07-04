def get_note_index(note):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return notes.index(note)

def get_note_from_index(index):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return notes[index % 12]

def get_note_with_octave(note, base_octave, interval):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    base_index = get_note_index(note)
    new_index = base_index + interval
    
    # 計算新的八度數
    octave_change = new_index // 12
    final_octave = base_octave + octave_change
    
    # 取得新音符
    new_note = notes[new_index % 12]
    return f"{new_note}-{final_octave}"

def get_chord_notes(chord_name):
    # 解析和弦名稱
    parts = chord_name.split('-')
    
    if len(parts) == 3:  
        root = parts[0]
        octave = int(parts[1])
        chord_type = parts[2]
    else:  
        if len(parts) == 2:
            root = parts[0]
            chord_type = parts[1]
        else:
            root = chord_name[0]
            if len(chord_name) > 1 and chord_name[1] in ['#']:
                root += '#'
                chord_type = chord_name[2:]
            else:
                chord_type = chord_name[1:]
        octave = 4  
    
    if not chord_type:
        chord_type = 'maj'  
        
    # 定義各種和弦的音程間隔
    chord_intervals = {
        'maj': [0, 4, 7],          
        'min': [0, 3, 7],          
        'dim': [0, 3, 6],         
        'aug': [0, 4, 8],         
        'maj7': [0, 4, 7, 11],     
        'min7': [0, 3, 7, 10],    
        'dom7': [0, 4, 7, 10]      
    }
    
    if chord_type not in chord_intervals:
        return f"不支援的和弦類型: {chord_type}"
    
    # 計算和弦音
    intervals = chord_intervals[chord_type]
    chord_notes = [get_note_with_octave(root, octave, interval) for interval in intervals]
    
    # 返回和弦資訊
    chord_type_names = {
        'maj': '大三和弦',
        'min': '小三和弦',
        'dim': '減三和弦',
        'aug': '增三和弦',
        'maj7': '大七和弦',
        'min7': '小七和弦',
        'dom7': '屬七和弦'
    }
    
    return f"{chord_name} ({chord_type_names[chord_type]}): {' '.join(chord_notes)}"

def main():
    print("鋼琴和弦查詢程式")
    print("支援的和弦類型: maj(大三和弦), min(小三和弦), dim(減三和弦), aug(增三和弦)")
    print("              maj7(大七和弦), min7(小七和弦), dom7(屬七和弦)")
    print("輸入格式範例: C-4-maj, F#-3-min7, Dmaj7")
    print("若不指定音高，預設為第4個八度")
    print("輸入 'q' 來結束程式")
    
    while True:
        chord = input("\n請輸入和弦名稱: ")
        if chord.lower() == 'q':
            break
        print(get_chord_notes(chord))

if __name__ == "__main__":
    main()