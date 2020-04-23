import json


def serialize_json(key, val):
    try:
        with open("{}.json".format(key), "w") as f:
            json.dump(val, f)
    except:
        return False
    return True
