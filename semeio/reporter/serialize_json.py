import json


def serialize_json(key, val):
    try:
        with open("{}.json".format(key), "w") as f:
            json.dump(val, f)
    except OverflowError:
        # Happens when there is a cycle
        return False
    except TypeError:
        # Unserialisable type
        return False
    return True
