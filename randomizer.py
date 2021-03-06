from randomtools.tablereader import (
        TableObject, get_global_label, tblpath, addresses, get_random_degree,
        get_activated_patches, mutate_normal, shuffle_normal, write_patch)
from randomtools.utils import (
    cached_property, classproperty, get_snes_palette_transformer,
    utilrandom as random)
from randomtools.interface import (
    get_outfile, get_seed, get_flags, get_activated_codes,
    run_interface, rewrite_snes_meta, clean_and_write, finish_interface)
from collections import defaultdict
from os import path, remove


VERSION = 7
ALL_OBJECTS = None


NAMES_PATH = path.join(tblpath, "item_names.txt")
JOB_NAMES_PATH = path.join(tblpath, "job_names.txt")
ITEM_NAMES, JOB_NAMES  = {}, []

with open(NAMES_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        while "  " in line:
            line = line.replace("  ", " ")
        if ' ' in line:
            index, name = line.split(' ', 1)
            index = int(index, 0x10)
            assert index not in ITEM_NAMES
            ITEM_NAMES[index] = name

with open(JOB_NAMES_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        JOB_NAMES.append(line)


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


class JobCrystalObject(TableObject):
    flag = "y"
    flag_description = "jobs obtained from crystal shards"
    custom_random_enable = True

    @classproperty
    def after_order(cls):
        return [JobAbilityObject, JobCommandObject]

    @classmethod
    def map_crystal_job(cls, value):
        if "GBA" in get_global_label():
            try:
                return {27: 20,
                        26: 21,
                        25: 22,
                        24: 23,
                        19: 24,
                        18: 25}[value]
            except KeyError:
                pass
        return (7-(value%8)) + ((value/8)*8)

    @property
    def name(self):
        if 'GBA' not in get_global_label() and self.job_index >= 20:
            return JOB_NAMES[self.job_index+4]
        return JOB_NAMES[self.job_index]

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
    def job_abilities(self):
        try:
            jaos = JobAbilityObject.groups[self.job_index]
        except KeyError:
            jaos = []
        return jaos

    @property
    def ability_ap_rank(self):
        jaos = self.job_abilities
        if not jaos:
            return 0
        rank = sum([jao.ap for jao in jaos]) / float(len(jaos))
        return rank

    @cached_property
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
        elif self.index <= 21:
            return 7
        elif self.index <= 24:
            return 8
        elif self.index == 25:
            return 9
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
        if 'GBA' not in get_global_label():
            assert len(candidates) == len(shuffled) == 21
        else:
            assert len(candidates) == len(shuffled) == 25
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

    def cleanup(self):
        if "GBA" in get_global_label() and self.is_freelancer:
            knight = [jco for jco in JobCrystalObject.every
                      if jco.crystal_index == 7]
            if knight:
                knight[0].crystal_index = self.crystal_index
            self.crystal_index = 7


class MonsterObject(TableObject):
    flag = "m"
    flag_description = "monster stats"
    custom_random_enable = True

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
    randomselect_attributes = [
        "agility", "strength", "attack_multiplier", "evasion", "defense",
        "magic_defense", "magic_evasion", "hp", "mp", "xp", "gp",
        "level",
        ]
    magic_mutate_bit_attributes = {
        ("elemental_immunities", "absorptions", "weaknesses"): (
            0xFF, 0xFF, 0xFF),
        ("status_immunities", "status", "cant_evade", "command_immunity",
            "creature_type"): (0xFCFFFF, 0xFCFFFF, 0xFF, 0x98, 0xFF),
        }

    @property
    def drops(self):
        drops = DropObject.get(self.index)
        return (drops.steal_rare, drops.steal_common,
                drops.drop_rare, drops.drop_common)

    @cached_property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank

        if 'GBA' in get_global_label():
            BANNED_INDEXES = [0x50]
        else:
            BANNED_INDEXES = [0x150]
        if self.index in BANNED_INDEXES:
            self._rank = -1
            return self.rank

        for m in MonsterObject.every:
            m.is_boss
            m._rank = -1

        candidates = [m for m in MonsterObject.every
                      if m.index not in BANNED_INDEXES and m.hp and m.level]
        sorted_hp = sorted(candidates,
                           key=lambda m: (m.hp, m.level, m.signature))
        sorted_level = sorted(candidates,
                           key=lambda m: (m.level, m.hp, m.signature))
        candidates = sorted(candidates,
            key=lambda m: (max(sorted_hp.index(m),
                               sorted_level.index(m)), m.signature))

        for m in candidates:
            m._rank = candidates.index(m)

        return self.rank

    @property
    def intershuffle_valid(self):
        if self.rank <= 0:
            return False
        if not (self.level or self.hp):
            return False
        return not self.is_boss

    @property
    def mutate_valid(self):
        return self.rank > 0

    @cached_property
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

        assert (self.status_immunities & 0x30000 ==
                self.old_data["status_immunities"] & 0x30000)


class DropObject(TableObject):
    flag = "t"
    custom_random_enable = True

    randomselect_attributes = [
        ("steal_common", "steal_rare"),
        ("drop_common", "drop_rare"),
        ]

    @cached_property
    def rank(self):
        return MonsterObject.get(self.index).rank

    @property
    def intershuffle_valid(self):
        return MonsterObject.get(self.index).intershuffle_valid

    def mutate(self):
        for attr in ["steal_common", "steal_rare",
                      "drop_common", "drop_rare"]:
            value = getattr(self, attr)
            if value > 0 and PriceObject.get(value).rank >= 0:
                candidates = [p for p in PriceObject.every
                              if p.is_valid_treasure]
                chosen = PriceObject.get(value).get_similar(candidates)
                setattr(self, attr, chosen.index)

        if random.random() < self.random_degree:
            self.steal_rare, self.drop_rare = (self.drop_rare,
                                               self.steal_rare)


class PriceMixin(TableObject):
    flag = "p"
    custom_random_enable = True

    @classproperty
    def mutate_attributes(cls):
        if hasattr(cls.every[0], "significand"):
            return {"significand": (1, 0xFF)}
        elif hasattr(cls.every[0], "price_value"):
            return {"price_value": None}

    @property
    def price(self):
        if hasattr(self, "exponent"):
            return self.significand * (10**(self.exponent & 0x7))
        else:
            return self.price_value

    def cleanup(self):
        if hasattr(self, "exponent"):
            if (self.exponent & 7) >= 5:
                return
            while self.price > 65000:
                self.significand -= 1


class PriceObject(PriceMixin):
    BANNED_INDEXES = ([0x00, 0x01, 0xF7, 0xF8] +
                      range(0x6F, 0x81) + range(0xD1, 0xE0))

    @cached_property
    def rank(self):
        if self.index in self.BANNED_INDEXES or self.index not in ITEM_NAMES:
            return -1
        elif self.is_event_only_item:
            rank = max([p.price for p in PriceObject.every]) + 2
        elif self.price <= 2:
            rank = max([p.price for p in PriceObject.every]) + 1
        else:
            rank = self.price
        return rank

    @cached_property
    def is_event_only_item(self):
        for t in TreasureObject.every:
            if t.is_item and t.value == self.index:
                return False
        for s in ShopObject.every:
            if s.pretty_shop_type == "Magic":
                continue
            if self.index in s.items:
                return False
        for d in DropObject.every:
            for attr in ["steal_common", "steal_rare",
                          "drop_common", "drop_rare"]:
                if getattr(d, attr) == self.index:
                    return False
        return True

    @property
    def name(self):
        if self.index in ITEM_NAMES:
            return ITEM_NAMES[self.index]
        else:
            return "%x" % self.index

    @property
    def is_valid_treasure(self):
        if self.rank <= 0:
            return False
        return True


class SpellPriceObject(PriceMixin):
    @cached_property
    def rank(self):
        return self.price


class ShopObject(TableObject):
    flag = "p"
    flag_description = "shops"
    custom_random_enable = True

    @cached_property
    def rank(self):
        if self.shop_type == 0:
            prices = [SpellPriceObject.get(i).rank for i in self.items if i]
        else:
            prices = [PriceObject.get(i).rank for i in self.items if i]
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
        ShopObject.class_reseed("shops")
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

            if pretty_shop_type == "Magic":
                all_items = [SpellPriceObject.get(i) for i in all_items]
            else:
                all_items = [PriceObject.get(i) for i in all_items]
            all_items = sorted(all_items, key=lambda i: (i.rank, i.signature))
            done_items = set([])
            random.shuffle(shops)
            for s in shops:
                s.reseed("wares")
                shop_items = [i for i in s.items if i > 0]
                chosen_items = []
                while len(chosen_items) < len(shop_items):
                    base_index = random.choice(shop_items)
                    candidates = [i for i in all_items if i not in done_items]
                    if not candidates:
                        candidates = list(all_items)
                    candidates = [c for c in candidates
                                  if c not in chosen_items]
                    chosen = PriceObject.get(base_index).get_similar(
                        candidates, override_outsider=True)
                    chosen_items.append(chosen)
                assert len(chosen_items) == len(set(chosen_items))
                chosen_items = [c.index for c in chosen_items]
                s.items = chosen_items

    def cleanup(self):
        if hasattr(self, "zero"):
            assert self.zero == 0
        while len(self.items) < 8:
            self.items.append(0)
        assert len(self.items) == 8


class TreasureObject(TableObject):
    flag = "t"
    flag_description = "treasure"
    custom_random_enable = True

    intershuffle_attributes = [("treasure_type", "value")]

    @property
    def intershuffle_valid(self):
        if self.is_monster:
            return "miab" in get_activated_codes()
        return True

    @property
    def mutate_valid(self):
        if not self.intershuffle_valid:
            return False
        if self.is_monster or self.is_magic:
            return False
        if self.is_gold or self.is_item:
            return True
        return False

    @cached_property
    def rank(self):
        if not self.intershuffle_valid:
            return -1
        if self.is_magic:
            price = SpellPriceObject.get(self.value).rank
        elif self.is_item:
            price = PriceObject.get(self.value).rank
        elif self.is_monster:
            return random.randint(0, max(i.rank for i in PriceObject.every
                                         if not i.is_magic))
        else:
            price = self.value * (10**(self.treasure_type & 0x7))
        if not self.mutate_valid and not self.is_magic:
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

    def mutate(self):
        if not self.mutate_valid:
            return
        price = self.rank
        if self.treasure_type == 0x40:
            assert self.treasure_type == 0x40
            assert self.is_item
            assert not self.is_magic
            price = max(min(price, 65000), 1)
            candidates = [p for p in PriceObject.every if p.is_valid_treasure]
            chosen = PriceObject.get(self.value).get_similar(candidates)
            self.value = chosen.index
        else:  # gold
            assert self.is_gold

    def cleanup(self):
        assert self.x == self.old_data["x"]
        assert self.y == self.old_data["y"]
        assert not ((self.is_monster and (self.is_item or self.is_gold))
                    or (self.is_item and self.is_gold))


class JobAbilityObject(TableObject):
    flag = "a"
    flag_description = "job learned abilities"
    custom_random_enable = True

    mutate_attributes = {"ap": (1, 999)}
    intershuffle_attributes = ["ap"]

    @property
    def name(self):
        try:
            return JOB_NAMES[self.index]
        except IndexError:
            return "%x" % self.index

    @classproperty
    def every(cls):
        if hasattr(cls, "_every"):
            return cls._every
        cls._every = super(JobAbilityObject, cls).every
        mimic = JobAbilityObject(
            get_outfile(), addresses.mime_abilities,
            index=len(JobAbilityObject.every),
            groupindex=max(JobAbilityObject.groups)+1)
        cls._every.append(mimic)
        return cls.every

    @cached_property
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
        return sorted(jaos, key=lambda jao: (jao.ap, jao.index))

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
        if self.index in JobAbilityObject.groups:
            jao = JobAbilityObject.groups[self.index]
            self.count = len(jao)
            assert self.count <= 7


class JobStatsObject(TableObject):
    flag = "s"
    flag_description = "job stats"
    custom_random_enable = True

    mutate_attributes = {
        "strength": None,
        "agility": None,
        "stamina": None,
        "magic": None,
        }


class JobEquipObject(TableObject):
    flag = "q"
    flag_description = "job equippable items"
    custom_random_enable = True

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
    custom_random_enable = True

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
            if "fight" in get_activated_codes() and ability == 5:
                pass
            elif ability > 0 and random.random() <= (
                    self.random_degree ** 0.75):
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

        for i, c in enumerate(self.commands):
            if c in [0x25]:
                self.commands[i] = self.old_data["commands"][i]

        assert len(self.commands) == 4
        assert self.commands[2] == 0
        assert self.commands[0] > 0


class JobInnatesObject(TableObject):
    def cleanup(self):
        if "GBA" not in get_global_label():
            self.innates |= 0x8
        if "zerker" in get_activated_codes():
            self.innates |= 0x800


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
        if hasattr(self, "color19"):
            valid_color_indexes = [0, 6, 7] + range(11, 20)
        else:
            valid_color_indexes = [0, 6, 7] + range(11, 16)
        colors = [getattr(self, "color%s" % i) for i in valid_color_indexes]
        newcolors = t(colors)
        for i, color in zip(valid_color_indexes, newcolors):
            setattr(self, "color%s" % i, color)


def do_fiesta():
    print "Please enter a number to choose an option:"
    print "1. Choose my own jobs."
    print "2. Randomize my jobs."

    while True:
        x = int(raw_input("? "))
        if x in [1, 2]:
            break

    print
    if x == 1:
        print "JOB LIST"
        namestrs = ["{0:>2}. {1}".format(jco.job_index+1, jco.name)
                    for jco in
                    sorted(JobCrystalObject.every, key=lambda j: j.job_index)]
        maxlen = max(map(len, namestrs))+2
        num_per_row = 79 / maxlen
        num_rows = len(namestrs) / float(num_per_row)
        if num_rows != int(num_rows):
            num_rows += 1
        num_rows = int(num_rows)
        ss = ["" for _ in xrange(int(num_rows))]
        while namestrs:
            namestr = namestrs.pop(0)
            s = ss.pop(0)
            s += ("{0:%s}" % maxlen).format(namestr)
            ss.append(s)
        for s in sorted(ss):
            print s
        print
        chosen_jobs = []
        for element in ["Wind", "Water", "Fire", "Earth"]:
            while True:
                x = raw_input("%s job? " % element)
                try:
                    x = int(x) - 1
                    assert x >= 0
                    x = [jco for jco in JobCrystalObject.every
                         if jco.job_index == x][0]
                    chosen_jobs.append(x)
                    break
                except:
                    pass
        print "Your chosen jobs are: " + ", ".join(
            jco.name for jco in chosen_jobs)
        print
    else:
        JobCrystalObject.class_reseed("fiesta1")
        fight_jobs = [jco for jco in JobCrystalObject.every
                      if jco.has_fight_command and jco.job_abilities]
        wind_job = random.choice(fight_jobs)
        candidates = [jco for jco in JobCrystalObject.every
                      if jco.job_abilities and jco is not wind_job]
        chosen_jobs = [wind_job] + random.sample(candidates, 3)

    assert len(set(chosen_jobs)) == 4
    JobCrystalObject.class_reseed("fiesta2")

    crystal_indexes = [jco.crystal_index for jco in chosen_jobs]
    for jco in JobCrystalObject.every:
        if jco.rank <= 2:
            jco.crystal_index = crystal_indexes[0]
        elif jco.rank <= 3:
            jco.crystal_index = crystal_indexes[1]
        elif jco.rank <= 5:
            jco.crystal_index = crystal_indexes[2]
        else:
            jco.crystal_index = crystal_indexes[3]

    NONELEMENTAL_SKILLS = [0x8, 0xc, 0xd, 0xe, 0x11, 0x12, 0x13, 0x15, 0x16,
                           0x17, 0x18, 0x2a, 0x2b, 0x5f, 0x61, 0x62, 0x90]
    ability_indexes = [a.ability for jco in chosen_jobs
                       for a in jco.job_abilities]
    if not set(ability_indexes) & set(NONELEMENTAL_SKILLS):
        print ("WARNING! These jobs will not learn a "
               "reliable source of non-elemental damage.")
        print("At least one of them has the Fight skill.")
        x = raw_input("Are you sure you want to continue? (y/n) ")
        if x and x.lower()[0] == 'n':
            remove(get_outfile())
            exit(0)


if __name__ == "__main__":
    try:
        print ('You are using the FF5 "Grand Cross" '
               'randomizer version %s.' % VERSION)
        ALL_OBJECTS = [g for g in globals().values()
                       if isinstance(g, type) and issubclass(g, TableObject)
                       and g not in [TableObject]]

        codes = {"zerker": ["zerker"],
                 "miab": ["miab"],
                 "fight": ["fight"],
                 "fiesta": ["fiesta"],
                }

        run_interface(ALL_OBJECTS, snes=True, codes=codes, custom_degree=True)

        for code in codes:
            if code in get_activated_codes():
                print "%s CODE ACTIVATED." % code.upper()

        if "fiesta" in get_activated_codes():
            do_fiesta()

        hexify = lambda x: "{0:0>2}".format("%x" % x)
        numify = lambda x: "{0: >3}".format(x)
        minmax = lambda x: (min(x), max(x))
        randomize_rng()
        clean_and_write(ALL_OBJECTS)

        if "GBA" not in get_global_label():
            rewrite_snes_meta("FF5-GC", VERSION, lorom=False)
        finish_interface()
    except Exception, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
