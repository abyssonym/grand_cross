from randomtools.tablereader import TableObject, get_global_label, tblpath
from randomtools.utils import (
    classproperty, mutate_normal, shuffle_bits, get_snes_palette_transformer,
    utilrandom as random)
from randomtools.interface import (
    get_outfile, get_seed, get_flags, run_interface, rewrite_snes_meta,
    clean_and_write, finish_interface)
from collections import defaultdict
from os import path


VERSION = 1
ALL_OBJECTS = None
CRYSTAL_ADDR_FILE = path.join(tblpath, "crystal_list.txt")
caf = open(CRYSTAL_ADDR_FILE)
CRYSTAL_ADDRS = [int(line.strip().split()[0], 0x10)
                 for line in caf.readlines()]
caf.close()


class JobAbilityObject(TableObject):
    mutate_attributes = {"ap": (1, 999)}
    intershuffle_attributes = ["ap"]

    @classproperty
    def every(cls):
        if hasattr(cls, "_every"):
            return cls._every
        cls._every = super(JobAbilityObject, cls).every
        mimic = JobAbilityObject(get_outfile(), 0x115429, index=99,
                                 groupindex=20)
        cls._every.append(mimic)
        return cls.every

    @property
    def rank(self):
        return self.ap


class AbilityCountObject(TableObject):
    def cleanup(self):
        jao = JobAbilityObject.groups[self.index]
        self.count = len(jao)


class JobStatsObject(TableObject):
    mutate_attributes = {
        "strength": None,
        "agility": None,
        "stamina": None,
        "magic": None,
        }


class JobEquipObject(TableObject): pass


class JobInnatesObject(TableObject):
    def cleanup(self):
        self.innates |= 0x8


class JobPaletteObject(TableObject):
    def mutate(self):
        # color1 - outline
        # color2 - sclera
        # color3 - eye color
        # color4 - flesh, main color
        # color5 - hair, main color
        # color8 - flesh, shading
        # color9 - hair, shading
        # color10 - hair, highlight
        t = get_snes_palette_transformer(middle=True)
        valid_color_indexes = [0, 6, 7] + range(11, 16)
        colors = [getattr(self, "color%s" % i) for i in valid_color_indexes]
        newcolors = t(colors)
        for i, color in zip(valid_color_indexes, newcolors):
            setattr(self, "color%s" % i, color)


def randomize_crystal_shards():
    f = open(get_outfile(), "r+b")
    values = []
    for addr in CRYSTAL_ADDRS:
        f.seek(addr)
        value = ord(f.read(1))
        values.append(value)
    random.shuffle(values)
    for addr, v in zip(CRYSTAL_ADDRS, values):
        f.seek(addr)
        f.write(chr(v))
    f.close()
    return values


if __name__ == "__main__":
    try:
        print ('You are using the FF5 '
               'randomizer version %s.' % VERSION)
        ALL_OBJECTS = [g for g in globals().values()
                       if isinstance(g, type) and issubclass(g, TableObject)
                       and g not in [TableObject]]
        run_interface(ALL_OBJECTS, snes=True)
        '''
        mimic = JobAbilityObject.groups[20]
        knight = JobAbilityObject.groups[0]
        for ja in mimic:
            ja.groupindex = 0
        for ja in knight:
            ja.groupindex = 20
        '''
        hexify = lambda x: "{0:0>2}".format("%x" % x)
        numify = lambda x: "{0: >3}".format(x)
        minmax = lambda x: (min(x), max(x))
        randomize_crystal_shards()
        clean_and_write(ALL_OBJECTS)
        rewrite_snes_meta("FF5-R", VERSION, megabits=32, lorom=False)
        finish_interface()
        import pdb; pdb.set_trace()
    except Exception, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
