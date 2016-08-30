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
    flag = "a"
    flag_description = "job learned abilities"
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

    @classmethod
    def full_randomize(cls):
        super(JobAbilityObject, cls).full_randomize()
        new_groups = defaultdict(list)
        jaos = list(JobAbilityObject.every)
        group_indexes = sorted(set([jao.groupindex for jao in jaos]))
        random.shuffle(jaos)
        while jaos:
            valid_groups = [i for i in group_indexes if not new_groups[i]]
            if not valid_groups:
                valid_groups = [i for i in group_indexes
                                if len(new_groups[i]) < 7]
            index = random.choice(valid_groups)
            jao = jaos.pop()
            jao.groupindex = index
            new_groups[index].append(jao)

    @classmethod
    def groupsort(cls, jaos):
        return sorted(jaos, key=lambda jao: (jao.rank, jao.index))


class AbilityCountObject(TableObject):
    @classproperty
    def after_order(self):
        return [JobAbilityObject]

    def cleanup(self):
        jao = JobAbilityObject.groups[self.index]
        self.count = len(jao)
        assert self.count <= 7


class JobStatsObject(TableObject):
    flag = "s"
    flag_description = "job stats"
    mutate_attributes = {
        "strength": None,
        "agility": None,
        "stamina": None,
        "magic": None,
        }


class JobEquipObject(TableObject):
    flag = "q"
    flag_description = "job equippable items"

    @classmethod
    def frequency(self, value):
        if hasattr(self, "_frequency"):
            return self._frequency[value]
        self._frequency = {}
        equip_jobs = [j for j in JobEquipObject.every
                      if j.equipment != 0xFFFFFFFF]
        for i in xrange(32):
            mask = 1 << i
            counter = 0
            for j in equip_jobs:
                if j.equipment & mask:
                    counter += 1
            self._frequency[i] = min(float(counter)/len(equip_jobs),
                                     1 / 3.0)
        return self.frequency(value)

    def mutate(self):
        if self.equipment == 0xFFFFFFFF:
            return
        for i in xrange(32):
            mask = 1 << i
            if random.random() < self.frequency(i):
                self.equipment ^= mask

    @classmethod
    def full_cleanup(cls):
        equippable = 0
        for j in cls.every:
            if j.equipment == 0xFFFFFFFF:
                continue
            equippable |= j.equipment
        for i in xrange(32):
            mask = (1 << i)
            if not mask & equippable:
                j = random.choice(cls.every)
                j.equipment |= mask
        super(JobEquipObject, cls).full_cleanup()


class JobCommandObject(TableObject):
    flag = "b"
    flag_description = "job base commands"

    @classproperty
    def after_order(self):
        return [JobAbilityObject]

    def randomize(self):
        if self.index == 6:
            candidates = [jao.ability for jao in
                          JobAbilityObject.groups[self.index]
                          if jao.ability > 0x4D]
            if candidates:
                self.commands[1] = random.choice(candidates)
            return

        if self.index > 20:
            return

        old_commands = list(self.commands)
        candidates = [jao.ability for jao in
                      JobAbilityObject.groups[self.index]
                      if jao.ability <= 0x4D]
        if self.index == 20:
            candidates += [5, 2]

        redundant_groups = [
                range(0x2C, 0x32),
                range(0x32, 0x38),
                range(0x38, 0x3E),
                range(0x3E, 0x44),
                ]
        for i, ability in enumerate(self.commands):
            if not candidates:
                break
            if ability > 0 and random.randint(1, 3) == 3:
                new_command = random.choice(candidates)
                if new_command in self.commands:
                    continue
                self.commands[i] = new_command
                for rg in redundant_groups:
                    if len(set(self.commands) & set(rg)) >= 2:
                        self.commands[i] = ability
                        break
                else:
                    candidates.remove(new_command)
            if ability in candidates:
                candidates.remove(ability)
        if not set(self.commands) & set([5, 0x2b, 2]):
            self.commands = old_commands
        for rg in redundant_groups:
            if len(set(self.commands) & set(rg)) >= 2:
                assert False

    def cleanup(self):
        if self.index < 20:
            cleaned_commands = [i for i in self.commands if i not in [5, 0x2b, 2]]
            if 0x2B in self.commands:
                cleaned_commands.insert(0, 0x2B)
            if 5 in self.commands:
                cleaned_commands.insert(0, 5)
            if 2 in self.commands:
                cleaned_commands.append(2)
            if cleaned_commands[3] == 0:
                cleaned_commands[3] = cleaned_commands[2]
                cleaned_commands[2] = 0
            self.commands = cleaned_commands
        assert len(self.commands) == 4
        assert self.commands[2] == 0
        assert self.commands[0] > 0


class JobInnatesObject(TableObject):
    # this flag stuff is a hack
    # because I'm too lazy to make a way to add flags manually
    flag = "s"
    flag_description = "jobs obtained from crystal shards"

    def cleanup(self):
        self.innates |= 0x8


class JobPaletteObject(TableObject):
    flag = "c"
    flag_description = "job outfit colors"

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
    if "s" not in get_flags():
        return
    f = open(get_outfile(), "r+b")
    values = []
    # galuf has no mime sprite
    addrs = [a for a in CRYSTAL_ADDRS if a != 0x91baf]
    for addr in addrs:
        f.seek(addr)
        value = ord(f.read(1))
        values.append(value)
    random.shuffle(values)
    for addr, v in zip(addrs, values):
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
        hexify = lambda x: "{0:0>2}".format("%x" % x)
        numify = lambda x: "{0: >3}".format(x)
        minmax = lambda x: (min(x), max(x))
        randomize_crystal_shards()
        clean_and_write(ALL_OBJECTS)
        rewrite_snes_meta("FF5-R", VERSION, lorom=False)
        finish_interface()
        import pdb; pdb.set_trace()
    except Exception, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
