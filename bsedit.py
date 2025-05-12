import json
from enum import Enum

class cutDirection(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UPLEFT = 4
    UPRIGHT = 5
    DOWNLEFT = 6
    DOWNRIGHT = 7
    DOT = 8

class colorIndex(Enum):
    LEFT = 0
    RIGHT = 1

class Note():
    def __init__(self, x, y, color_index, cut_direction, angle_offset=0):
        self.x = x
        self.y = y
        self.color_index = color_index,
        self.cut_direction = cut_direction,
        self.angle_offset = angle_offset

    def __call__(self):
        output = {
            "x": self.x,
            "y": self.y,
            "color_index": self.color_index,
            "cut_direction": self.cut_direction,
            "angle_offset": self.angle_offset
        }
        return output
    
class Arch():
    def __init__(self, start_beat, start_note, end_beat, end_note, start_length_multiplier=1.0, end_length_multiplier=1.0, mid_anchor_mode=0):
        self.start_beat = start_beat
        self.color_index = start_note.color_index
        self.start_x = start_note.x
        self.start_y = start_note.y
        self.start_direction = start_note.cut_direction
        self.start_length_multiplier = start_length_multiplier
        self.end_beat = end_beat
        self.end_x = end_note.x
        self.end_y = end_note.y
        self.end_direction = end_note.cut_direction
        self.end_length_multiplier = end_length_multiplier
        self.mid_anchor_mode = mid_anchor_mode
    
    def __init__(self, c, b, x, y, d, mu, tb, tx, ty, tc, tmu, m):
        self.start_beat = b
        self.color_index = colorIndex(c)
        self.start_x = x
        self.start_y = y
        self.start_direction = cutDirection(d)
        self.start_length_multiplier = mu
        self.end_beat = tb
        self.end_x = tx
        self.end_y = ty
        self.end_direction = cutDirection(tc)
        self.end_length_multiplier = tmu
        self.mid_anchor_mode = m

class ChainNote():
    def __init__(self, x, y, color_index, cut_direction, duration, end_x, end_y, slice_count, squish_factor=0.5):
        self.x = x
        self.y = y
        self.color_index = color_index
        self.cut_direction = cut_direction
        self.duration = duration
        self.end_x = end_x
        self.end_y = end_y
        self.slice_count = slice_count
        self.squish_factor = squish_factor

    def __init__(self, c, b, x, y, d, tb, tx, ty, sc, s):
        self.x = x
        self.y = y
        self.color_index = colorIndex(c)
        self.cut_direction = cutDirection(d)
        self.duration = tb - b
        self.end_x = tx
        self.end_y = ty
        self.slice_count = sc
        self.squish_factor = s

class Bomb():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self):
        output = {
            "x": self.x,
            "y": self.y,
        }
        return output
    
class Wall():
    def __init__(self, x, y, duration, width, height):
        self.x = x,
        self.y = y,
        self.duration = duration,
        self.width = width,
        self.height = height

    def __call__(self):
        output = {
            "x": self.x,
            "y": self.y,
            "duration": self.duration,
            "width": self.width,
            "height": self.height
        }
        return output

class Level():
    def __init__(self, file_path=None):
        
        self.notes = []
        self.bombs = []
        self.walls = []
        self.arches = []
        self.chain_notes = []
        self.bpm_changes = []

        if file_path == None:
            return
        
        with open(file_path, 'r') as f:
            raw_data = json.load(f)

        for note in raw_data["colorNotes"]:
            new_note = (note["b"], Note(note["x"], note["y"], colorIndex(note["c"]), cutDirection(note["d"])))
            self.notes.append(new_note)
        
        for bomb in raw_data["bombNotes"]:
            new_bomb = (bomb["b"], Bomb(bomb["x"], bomb["y"]))
            self.bombs.append(new_bomb)
        
        for wall in raw_data["obstacles"]:
            new_wall = (wall["b"], Wall(wall["x"], wall["y"], wall["d"], wall["w"], wall["h"]))
            self.walls.append(new_wall)

        for arch in raw_data["sliders"]:
            new_arch = Arch(arch["c"], arch["b"], arch["x"], arch["y"], arch["d"], arch["mu"], arch["tb"], arch["tx"], arch["ty"], arch["tc"], arch["tmu"], arch["m"])
            self.arches.append(new_arch)
        
        for chain_note in raw_data["burstSliders"]:
            new_chain_note = (chain_note["b"], ChainNote(chain_note["c"], chain_note["b"], chain_note["x"], chain_note["y"], chain_note["d"], chain_note["tb"], chain_note["tx"], chain_note["ty"], chain_note["sc"], chain_note["s"]))
            self.chain_notes.append(new_chain_note)

    def add_note(self, beat, note):
        self.notes.append((beat, note))
    def add_bomb(self, beat, bomb):
        self.bombs.append((beat, bomb))
    def add_wall(self, beat, wall):
        self.walls.append((beat, wall))
    def add_arch(self, arch):
        self.arches.append(arch)
    def add_chain_note(self, beat, chain_note):
        self.notes.append((beat, chain_note))

    def clear_notes(self):
        self.notes = []
    def clear_bombs(self):
        self.bombs = []
    def clear_walls(self):
        self.walls = []
    def clear_arches(self):
        self.arches = []
    def clear_chain_notes(self):
        self.chain_notes = []
    

    def compile(self, output_path):
        with open("blank.dat", 'r') as f:
            raw_data = json.load(f)
        
        for noteTuple in self.notes:
            note = noteTuple[1]
            beat = noteTuple[0]

            output = {
                "b": beat,
                "x": note.x,
                "y": note.y,
                "a": note.angle_offset,
                "c": note.color_index[0].value,
                "d": note.cut_direction[0].value
            }
            raw_data["colorNotes"].append(output)
        
        for bombTuple in self.bombs:
            bomb = bombTuple[1]
            beat = bombTuple[0]
            
            output = {
                "b": beat,
                "x": bomb.x,
                "y": bomb.y,
            }
            raw_data["bombNotes"].append(output)
        
        for wallTuple in self.walls:
            wall = wallTuple[1]
            beat = wallTuple[0]
            
            output = {
            "b": beat,
            "x": wall.x,
            "y": wall.y,
            "d": wall.duration,
            "w": wall.width,
            "h": wall.height
            }
            raw_data["obstacles"].append(output)

        for arch in self.arches:
            
            output = {
            "c": arch.color_index.value,
            "b": arch.start_beat,
            "x": arch.start_x,
            "y": arch.start_y,
            "d": arch.start_direction.value,
            "mu": arch.start_length_multiplier,
            "tb": arch.end_beat,
            "tx": arch.end_x,
            "ty": arch.end_y,
            "tc": arch.end_direction.value,
            "tmu": arch.end_length_multiplier,
            "m": arch.mid_anchor_mode
            }
            raw_data["sliders"].append(output)

        for chain_note_tuple in self.chain_notes:

            beat = chain_note_tuple[0]
            chain_note = chain_note_tuple[1]
            
            output = {
            "c": chain_note.color_index.value,
            "b": beat,
            "x": chain_note.x,
            "y": chain_note.y,
            "d": chain_note.cut_direction.value,
            "tb": beat + chain_note.duration,
            "tx": chain_note.end_x,
            "ty": chain_note.end_y,
            "sc": chain_note.slice_count,
            "s": chain_note.squish_factor,
            }
            raw_data["burstSliders"].append(output)
        
        with open(output_path, 'w') as f:
            f.write(json.dumps(raw_data))


        
        

        
            



