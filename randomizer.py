from randomtools.tablereader import (
        TableObject, get_global_label, tblpath, addresses, get_random_degree,
        get_activated_patches, mutate_normal, shuffle_normal, write_patch)
from randomtools.utils import (
    classproperty, get_snes_palette_transformer, utilrandom as random)
from randomtools.interface import (
    get_outfile, get_seed, get_flags, run_interface, rewrite_snes_meta,
    clean_and_write, finish_interface)
from collections import defaultdict
from os import path


VERSION = 4
ALL_OBJECTS = None
CRYSTAL_ADDR_FILE = path.join(tblpath, "crystal_list.txt")
caf = open(CRYSTAL_ADDR_FILE)
CRYSTAL_ADDRS = [int(line.strip().split()[0], 0x10)
                 for line in caf.readlines()]
caf.close()


def shuffle_bits(value, size=8):
    numbits = bin(value).count("1")
    if numbits:
        digits = random.sample(range(size), numbits)
        newvalue = 0
        for d in digits:
            newvalue |= (1 << d)
        value = newvalue
    return value


def divisibility_rank(level):
    scores = {2: 5000,
              3: 100,
              4: 1000,
              5: 10000}
    remaining = scores.keys()
    rank = 0
    halves = 0
    while remaining and level > 0:
        for divisor in list(remaining):
            if not level % divisor:
                rank += (scores[divisor] * (0.5**halves))
                remaining.remove(divisor)
        halves += 1
        level /= 2
    return rank


def randomize_rng():
    filename = get_outfile()
    f = open(filename, "r+b")
    f.seek(addresses.rng)
    random_numbers = range(0x100)
    random.shuffle(random_numbers)
    f.write("".join(map(chr, random_numbers)))
    f.close()


def item_is_buyable(value, magic=False):
    if magic:
        shops = ShopObject.get_magic_shops()
    else:
        shops = ShopObject.get_nonmagic_shops()
    for s in shops:
        if value in s.items:
            return True
    else:
        return False


def get_item_similar_price(price=None, magic=False):
    if magic:
        shops = ShopObject.get_magic_shops()
    else:
        shops = ShopObject.get_nonmagic_shops()

    items = set([])
    for s in shops:
        items |= set(s.items)

    if magic:
        items = sorted([(PriceObject.get(i+0x100).price, i) for i in items])
    else:
        items = sorted([(PriceObject.get(i).price, i) for i in items])

    newprice = mutate_normal(price, minimum=1, maximum=max(65000, price),
                             random_degree=TreasureObject.random_degree)
    items = items[:1] + [i for (p, i) in items if p <= newprice]
    chosen = items[-1]
    return chosen


class JobCrystalObject(TableObject):
    flag = "y"
    flag_description = "jobs obtained from crystal shards"

    @classproperty
    def after_order(cls):
        return [JobAbilityObject, JobCommandObject]

    @classmethod
    def map_crystal_job(cls, value):
        return (7-(value%8)) + ((value/8)*8)

    @property
    def job_index(self):
        return self.map_crystal_job(self.crystal_index)

    @property
    def is_freelancer(self):
        return self.pointer == addresses.freelancer

    @property
    def is_mime(self):
        return self.pointer == addresses.mime

    @property
    def has_fight_command(self):
        return JobCommandObject.get(self.job_index).commands[0] == 5

    @property
    def intershuffle_valid(self):
        return not self.is_mime

    @property
    def ability_ap_rank(self):
        if self.is_freelancer:
            return 0
        jaos = JobAbilityObject.groups[self.job_index]
        rank = sum([jao.ap for jao in jaos]) / float(len(jaos))
        return rank

    @property
    def rank(self):
        if self.index == 0:
            return 1
        elif self.index <= 6:
            return 2
        elif self.index <= 9:
            return 4
        elif self.index <= 13:
            return 6
        elif self.index <= 15:
            return 5
        elif self.index <= 20:
            return 3
        elif self.index == 21:
            return 7
        raise Exception("Unknown index.")

    @classmethod
    def intershuffle(cls):
        candidates = sorted(
            [jco for jco in JobCrystalObject.every if jco.intershuffle_valid],
            key=lambda jco: (jco.ability_ap_rank, jco.signature))
        shuffled = []
        while candidates:
            max_index = len(candidates)-1
            index = random.randint(0, max_index)
            degree = JobCrystalObject.random_degree ** 0.25
            if degree <= 0.5:
                degree = degree * 2
                a, b = 0, index
            else:
                degree = (degree - 0.5) * 2
                a, b = index, max_index
            index = int(round((a*(1-degree)) + (b*degree)))
            index = random.randint(0, index)
            chosen = candidates[index]
            shuffled.append(chosen)
            candidates.remove(chosen)

        candidates = sorted(
            shuffled, key=lambda jco: (jco.rank, jco.signature))
        assert len(candidates) == len(shuffled) == 21
        for c, s in zip(candidates, shuffled):
            c.crystal_index = s.old_data["crystal_index"]

        freelancer = [jco for jco in candidates if jco.is_freelancer][0]
        fight_crystals = [jco for jco in shuffled if jco.has_fight_command]
        if freelancer not in fight_crystals:
            assert not freelancer.has_fight_command
            chosen = fight_crystals[0]
            freelancer.crystal_index, chosen.crystal_index = (
                chosen.crystal_index, freelancer.crystal_index)
        assert freelancer.has_fight_command


class MonsterObject(TableObject):
    flag = "m"
    flag_description = "monster stats"
    mutate_attributes = {
        "agility": None,
        "strength": None,
        "attack_multiplier": None,
        "evasion": None,
        "defense": None,
        "magic": None,
        "magic_defense": None,
        "magic_evasion": None,
        "hp": (1, 65535),
        "mp": (1, 65535),
        "xp": (1, 65535),
        "gp": (1, 65535),
        }
    intershuffle_attributes = [
        "agility", "evasion", "defense",
        "magic_defense", "magic_evasion", "hp", "mp", "xp", "gp",
        "level",
        ]
    magic_mutate_bit_attributes = {
        ("elemental_immunities", "absorptions", "weaknesses"): (
            0xFF, 0xFF, 0xFF),
        ("status_immunities", "status"): (0xFFFFFF, 0xFDFFFF),
        ("cant_evade", "command_immunity"): (0xFF, 0x98),
        }

    @property
    def drops(self):
        drops = DropObject.get(self.index)
        return (drops.steal_rare, drops.steal_common,
                drops.drop_rare, drops.drop_common)

    @property
    def rank(self):
        factors = [
            "level",
            "hp",
            ["strength", "magic"],
            ["defense", "magic_defense"],
            ["evasion", "magic_evasion"],
            ]
        rank = 1
        for factor in factors:
            if not isinstance(factor, list):
                factor = [factor]
            factor = [getattr(self, attr) for attr in factor]
            factor = max(factor + [1])
            rank *= factor
        return rank

    @property
    def intershuffle_valid(self):
        if not (self.level or self.hp):
            return False
        return not (self.is_boss or self.level < 10)

    @property
    def is_boss(self):
        return self.get_bit("heavy") or (
            self.get_bit("control") and self.get_bit("catch"))

    def mutate(self):
        super(MonsterObject, self).mutate()

        if 1 <= self.level <= 99:
            new_level = mutate_normal(self.level, minimum=1, maximum=99,
                                      random_degree=self.random_degree)
            old_divisibility = divisibility_rank(self.level)
            new_divisibility = divisibility_rank(new_level)
            if new_divisibility < old_divisibility:
                if not self.is_boss:
                    self.level = new_level
                else:
                    difference = float(new_level) / self.level
                    if difference > 1:
                        difference = 1 / difference
                    difference = (difference * (1-self.random_degree)) + (
                        self.random_degree**2)
                    if random.random() < difference:
                        self.level = new_level
            elif (not self.is_boss
                    and random.random() < (self.random_degree ** 0.5)):
                self.level = new_level

    def cleanup(self):
        self.creature_type = self.old_data["creature_type"]

        if self.is_boss:
            for attr in self.mutate_attributes:
                if getattr(self, attr) < self.old_data[attr]:
                    setattr(self, attr, self.old_data[attr])

        self.status ^= ((self.status & self.status_immunities)
            ^ (self.old_data["status"] & self.old_data["status_immunities"]))
        self.status |= (self.old_data["status"]
            & self.old_data["status_immunities"] & self.status_immunities)
        if self.status & (1 << 17) and not self.old_data["status"] & (1 << 17):
            # necrophobia invulnerability
            self.status ^= (1 << 17)

        self.elemental_immunities = self.elemental_immunities & (
            self.elemental_immunities ^ (self.absorptions | self.weaknesses))
        self.command_immunity &= 0x98

        if self.index == 0x8a:
            # Pao's tent drop
            d = DropObject.get(self.index)
            d.drop_common = d.old_data["drop_common"]


class DropObject(TableObject):
    flag = "t"
    intershuffle_attributes = [
        ("steal_common", "steal_rare"),
        ("drop_common", "drop_rare"),
        ]

    @property
    def rank(self):
        return MonsterObject.get(self.index).rank

    @property
    def intershuffle_valid(self):
        return MonsterObject.get(self.index).intershuffle_valid

    def mutate(self):
        for attr in ["steal_common", "steal_rare",
                      "drop_common", "drop_rare"]:
            value = getattr(self, attr)
            if value > 0 and item_is_buyable(value):
                price = PriceObject.get(value).price
                setattr(self, attr, get_item_similar_price(price))

        if random.choice([True, False]):
            if random.choice([True, False]):
                self.steal_common, self.drop_common = (self.drop_common,
                                                       self.steal_common)
            if random.choice([True, False]):
                self.steal_rare, self.drop_rare = (self.drop_rare,
                                                   self.steal_rare)

class PriceObject(TableObject):
    flag = "p"
    mutate_attributes = {"significand": (1, 0xFF)}

    @property
    def rank(self):
        return self.price

    @property
    def price(self):
        return self.significand * (10**(self.exponent & 0x7))

    def cleanup(self):
        if (self.exponent & 7) >= 5:
            return
        while self.price > 65000:
            self.significand -= 1


class ShopObject(TableObject):
    flag = "p"
    flag_description = "shops"

    @property
    def rank(self):
        if self.shop_type == 0:
            prices = [PriceObject.get(i+0x100).price for i in self.items if i]
        else:
            prices = [PriceObject.get(i).price for i in self.items if i]
        if not prices:
            return -1
        return max(prices) * sum(prices) / len(prices)

    @property
    def pretty_shop_type(self):
        shop_type = self.shop_type & 0x7
        types = {0: "Magic", 1: "Weapons", 2: "Armor", 3: "Items",
                 4: "Armor", 5: "Weapons", 6: "Items"}
        if shop_type in types:
            return types[shop_type]
        return self.shop_type

    @classmethod
    def get_magic_shops(cls):
        return [s for s in cls.every if s.pretty_shop_type == "Magic"]

    @classmethod
    def get_nonmagic_shops(cls):
        return [s for s in cls.every if s.pretty_shop_type != "Magic"]

    def __repr__(self):
        s = hex(self.index) + " %s %s" % (self.shop_type,
                                          self.pretty_shop_type)
        for i in self.items:
            if i > 0:
                if self.pretty_shop_type == "Magic":
                    pi = i + 0x100
                else:
                    pi = i
                s += "\n%x %s" % (i, PriceObject.get(pi).price)
        return s

    @classmethod
    def full_randomize(cls):
        super(ShopObject, cls).full_randomize()
        for pretty_shop_type in ["Magic", "Weapons", "Armor", "Items"]:
            shops = [s for s in ShopObject.ranked if s.rank > 0 and
                     s.pretty_shop_type == pretty_shop_type]
            itemranks = defaultdict(set)
            all_items = set([])
            avg = 0
            for n, s in enumerate(shops):
                items = [i for i in s.items if i]
                itemranks[n] |= set(items)
                all_items |= set(items)
                avg += len(items)

            avg /= float(len(shops))
            for n, s in enumerate(shops):
                num_items = int(round(mutate_normal(avg, minimum=1,
                                                    maximum=8)))
                chosen_items = set([])
                while len(chosen_items) < num_items:
                    chosen_rank = mutate_normal(n, minimum=0,
                                                maximum=len(shops)-1)
                    candidates = sorted([i for i in itemranks[chosen_rank]
                                         if i not in chosen_items])
                    if not candidates:
                        a = n
                        b = len(shops) / 2
                        a, b = min(a, b), max(a, b)
                        n = random.randint(a, b)
                        continue
                    item = random.choice(candidates)
                    chosen_items.add(item)
                s.items = sorted(chosen_items)
                all_items -= chosen_items

            all_items = sorted(all_items)
            while all_items:
                item = all_items.pop()
                candidates = [s for s in shops if len(s.items) < 8]
                s = random.choice(candidates)
                s.items.append(item)

    def cleanup(self):
        while len(self.items) < 8:
            self.items.append(0)
        assert len(self.items) == 8


class TreasureObject(TableObject):
    flag = "t"
    flag_description = "treasure"
    intershuffle_attributes = [("treasure_type", "value")]

    @property
    def intershuffle_valid(self):
        if self.is_monster:
            return False
        if self.treasure_type & 0x18:
            return False
        if self.is_gold and self.treasure_type & 0x7 > 4:
            return False
        if bin(self.treasure_type & 0xE0).count("1") > 1:
            return False
        if not self.is_gold and self.treasure_type & 0x7:
            return False
        return True

    @property
    def mutate_valid(self):
        if not self.intershuffle_valid:
            return False
        if self.is_gold:
            return True
        if self.is_magic:
            return item_is_buyable(self.value, magic=True)
        if self.is_item:
            return item_is_buyable(self.value, magic=False)

    @property
    def rank(self):
        if not self.intershuffle_valid:
            return -1
        if self.is_magic:
            price = PriceObject.get(self.value + 0x100).price
        elif self.is_item:
            price = PriceObject.get(self.value).price
        else:
            price = self.value * (10**(self.treasure_type & 0x7))
        if not self.mutate_valid:
            price += 60000
        return price

    @property
    def is_monster(self):
        return self.treasure_type & 0x80

    @property
    def is_item(self):
        return not self.is_monster and self.treasure_type & 0x40

    @property
    def is_magic(self):
        return self.treasure_type & 0x20 and not self.treasure_type & 0xC0

    @property
    def is_gold(self):
        return not self.treasure_type & 0xE0

    def cleanup(self):
        return
        print hex(self.index), hex(self.pointer)
        assert not ((self.is_monster and (self.is_item or self.is_gold))
                    or (self.is_item and self.is_gold))

    def mutate(self):
        if not self.mutate_valid:
            return
        chance = random.random()
        price = self.rank
        if chance <= 0.85:
            if chance <= 0.70:  # item
                self.treasure_type = 0x40
            else:  # magic
                self.treasure_type = 0x20
            self.value = get_item_similar_price(price, magic=self.is_magic)
        else:  # gold
            price = mutate_normal(price, minimum=1, maximum=max(65000, price),
                                  random_degree=TreasureObject.random_degree)
            price = min(65000, max(100, price))
            exponent = 0
            while price >= 100:
                price /= 10
                exponent += 1
            self.treasure_type = exponent
            self.value = price


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
        mimic = JobAbilityObject(get_outfile(), addresses.mime_abilities,
                                 index=99, groupindex=20)
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

    def cleanup(self):
        if not (self.ap == 999 or self.ap < 5):
            ap = int(round(self.ap*2, -1)) / 2
            assert ap % 5 == 0
            assert abs(ap - self.ap) <= 4
            self.ap = ap
        self.ap = min(999, max(1, self.ap))


class AbilityCountObject(TableObject):
    @classproperty
    def after_order(self):
        if 'a' in get_flags():
            return [JobAbilityObject]
        return []

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

    magic_mutate_bit_attributes = {
        ("equipment",): (0xFFFFFFFF,)
        }

    @classmethod
    def full_cleanup(cls):
        equippable = 0
        all_mask = 0xFFFFFFFF
        none_mask = 0
        for j in cls.every:
            if j.equipment == 0xFFFFFFFF:
                continue
            equippable |= j.equipment
            all_mask &= j.old_data["equipment"]
            none_mask |= j.old_data["equipment"]

        for i in xrange(32):
            mask = (1 << i)
            if not mask & equippable:
                j = random.choice(cls.every)
                j.equipment |= mask

        for j in cls.every:
            j.equipment |= all_mask
            j.equipment &= none_mask

        super(JobEquipObject, cls).full_cleanup()

    def cleanup(self):
        if self.old_data["equipment"] == 0xFFFFFFFF:
            self.equipment = self.old_data["equipment"]


class JobCommandObject(TableObject):
    flag = "b"
    flag_description = "job base commands"

    @classproperty
    def after_order(self):
        if 'a' in get_flags():
            return [JobAbilityObject]
        return []

    def randomize(self):
        if self.index == 6:
            # berserker passive
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
            if ability > 0 and random.random() <= (self.random_degree ** 0.75):
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
        while not set(self.commands) & set([5, 0x2b, 2]):
            i, c = random.choice(sorted(enumerate(old_commands)))
            if c in [5, 0x2b, 2]:
                self.commands[i] = c
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


if __name__ == "__main__":
    try:
        print ('You are using the FF5 "Grand Cross" '
               'randomizer version %s.' % VERSION)
        ALL_OBJECTS = [g for g in globals().values()
                       if isinstance(g, type) and issubclass(g, TableObject)
                       and g not in [TableObject]]
        run_interface(ALL_OBJECTS, snes=True)
        hexify = lambda x: "{0:0>2}".format("%x" % x)
        numify = lambda x: "{0: >3}".format(x)
        minmax = lambda x: (min(x), max(x))
        randomize_rng()
        clean_and_write(ALL_OBJECTS)
        rewrite_snes_meta("FF5-GC", VERSION, lorom=False)
        finish_interface()
    except Exception, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
