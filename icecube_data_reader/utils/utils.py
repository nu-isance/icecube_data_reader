import numpy as np



class ddict(dict):
    """
    Modified dictionary class, derived from `dict`.
    Used to nest dictionaries without having to write [] all the time when adding or calling.
    """

    def __init__(self):
        super().__init__()

    def add(self, value, *keys):
        """
        Add value to chain of keys.
        Careful: this may overwrite existing values!
        :param value: Value to be added
        :param keys: Tuple containing ordered keys behind which the value should be added.
        """
        # TODO: protect from overwriting

        temp = self
        for key in keys[:-1]:
            try:
                temp[key]
            except KeyError:
                temp[key] = {}
            finally:
                temp = temp[key]
        temp[keys[-1]] = value

    def __call__(self, *keys):
        """
        Call value of nested dicts.
        :param keys: Tuple of ordered keys whose value should be returned
        :return: value behind tuple of keys
        """

        temp = self
        for key in keys:
            temp = temp[key]
        return temp
