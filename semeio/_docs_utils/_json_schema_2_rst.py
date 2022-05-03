from copy import deepcopy


def _insert_ref(schema):
    for value in schema.values():
        if isinstance(value, dict):
            _insert_ref(value)
        elif isinstance(value, list):
            for index, val in enumerate(value.copy()):
                if "$ref" in val:
                    value[index] = val["$ref"] + "_"


def _remove_key(schema, del_key):
    schema_copy = schema.copy()
    for key, val in schema_copy.items():
        if key == del_key:
            del schema[key]
        elif isinstance(val, dict):
            _remove_key(schema[key], del_key)


def _replace_key(schema, old_key, new_key):
    schema_copy = schema.copy()
    for key, val in schema_copy.items():
        if key == old_key:
            del schema[key]
            schema[new_key] = val
            key = new_key
        if isinstance(val, dict):
            _replace_key(schema[key], old_key, new_key)


def _create_docs(schema):
    """While there are alternatives to implementing something new, most of these
    rely on a sphinx directive, which we can not easily use here. There are
    also libraries such as jsonschema2rst, but they only document their
    command line interface, and the result is not immediately valid rst,
    and require some post-processing. If a good alternative is found,
    this could be removed."""
    schema = deepcopy(schema)
    _remove_key(schema, "title")
    _replace_key(schema, "anyOf", "must be one of")
    _replace_key(schema, "allOf", "must be")
    _replace_key(schema, "default", "default")
    _insert_ref(schema)
    return _make_documentation(schema)


def _make_documentation(
    schema, level=0, preface="", element_seperator="\n\n", reference=False
):
    indent = level * 4 * " "
    docs = []
    if isinstance(schema, dict):
        for key, val in schema.items():
            if key == "description":
                docs += [indent + val.replace("\n", " ")]
            elif isinstance(val, dict):
                if reference:
                    docs += [f".. _{key}:"]
                docs += [indent + f"**{key}**:"]
                docs += [
                    _make_documentation(
                        val, level=level + 1, reference=(key == "definitions")
                    )
                ]
            elif isinstance(val, list):
                docs += [indent + f"**{key}**:"]
                docs += [_make_documentation(val, level=level + 1)]
            else:
                docs += [indent + f"**{key}**: {val}"]
    elif isinstance(schema, list):
        list_docs = []
        for element in schema:
            list_docs += [
                _make_documentation(
                    element, level=level + 1, preface="* ", element_seperator="\n"
                )
            ]
        docs += list_docs
    else:
        docs += [indent + preface + schema]
    return element_seperator.join(docs)
