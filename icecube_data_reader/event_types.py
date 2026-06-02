"""
Organise different event types
"""

from dataclasses import dataclass


class Meta(type):
    # Construct to add str(EventType)
    def __repr__(self):
        return self._str

    # Construct to add str(EventType())

class EventType:
    def __str__(self):
        return self._str

    def __eq__(self, other):
        return self.S == other.S


@dataclass(eq=False)
class IC40(EventType, metaclass=Meta):
    _str = "IC40"
    S = 1


@dataclass(eq=False)
class IC59(EventType, metaclass=Meta):
    _str = "IC59"
    S = 2


@dataclass(eq=False)
class IC79(EventType, metaclass=Meta):
    _str = "IC79"
    S = 3


@dataclass(eq=False)
class IC86(EventType, metaclass=Meta):
    _str = "IC86"
    S = 4


@dataclass(eq=False)
class IC86_I(EventType, metaclass=Meta):
    _str = "IC86_I"
    S = 5


@dataclass(eq=False)
class IC86_II(EventType, metaclass=Meta):
    _str = "IC86_II"
    S = 6


# And what have they ever given us in return...
suffixes = [
    "_I",
    "_II",
    "_III",
    "_IV",
    "_V",
    "_VI",
    "_VII",
    "_VIII",
    "_IX",
    "_X",
    "_XI",
]


@dataclass
class DR2:
    available_irfs = [IC40, IC59, IC79, IC86]


@dataclass
class DR1:
    available_irfs = [IC40, IC59, IC79, IC86_I, IC86_II]

class Refrigerator:
    """Collect all event types"""
    
    detectors = [IC40, IC59, IC79, IC86, IC86_I, IC86_II]

    @classmethod
    def int2str(cls, int_):
        """Returns python event-type string corresponding to integer event-type"""

        for dm in cls.detectors:
            if int_ == dm._str:
                return dm._str
        else:
            raise ValueError(f"No detector {int_} available.")

    @classmethod
    def str2int(cls, str_):
        """Returns stan event-type corresponding to python event-type string"""

        for dm in cls.detectors:
            if str_ == str(dm):
                return dm.S
        else:
            raise ValueError(f"No detector {str_} available.")
        
    @classmethod
    def int2dm(cls, int_):

        for dm in cls.detectors:
            if int_ == dm.S:
                return dm
        else:
            raise ValueError(f"No detector {int_} available.")
        
    @classmethod
    def str2dm(cls, str_):
        return cls.int2dm(cls.str2int(str_))

