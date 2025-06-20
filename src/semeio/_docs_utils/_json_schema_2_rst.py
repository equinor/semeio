import textwrap
from copy import deepcopy


def _insert_ref(schema: dict, defs: dict) -> dict:
    schema_copy = schema.copy()
    for index, value in schema_copy.items():
        if isinstance(value, dict):
            schema[index] = _insert_ref(value, defs)
        elif isinstance(value, list):
            for index, val in enumerate(value.copy()):
                if isinstance(val, int):
                    val = str(val)
                if "$ref" in val:
                    value[index] = defs[val["$ref"]].pop("properties")
        elif index == "$ref":
            del schema["$ref"]
            schema["must be"] = defs[value].pop("properties")
    return schema


def _remove_key(schema: dict, del_key: str) -> None:
    schema_copy = schema.copy()
    for key, val in schema_copy.items():
        if key == del_key:
            del schema[key]
        elif isinstance(val, dict):
            _remove_key(schema[key], del_key)


def _replace_key(schema: dict, old_key: str, new_key: str) -> None:
    schema_copy = schema.copy()
    for key, val in schema_copy.items():
        if key == old_key:
            del schema[key]
            schema[new_key] = val
            key = new_key
        if isinstance(val, dict):
            _replace_key(schema[key], old_key, new_key)


def _create_docs(schema: dict) -> str:
    """While there are alternatives to implementing something new, most of these
    rely on a sphinx directive, which we can not easily use here. There are
    also libraries such as jsonschema2rst, but they only document their
    command line interface, and the result is not immediately valid rst,
    and require some post-processing. If a good alternative is found,
    this could be removed."""
    schema = deepcopy(schema)
    schema.pop("type")
    _remove_key(schema, "title")
    _replace_key(schema, "anyOf", "must be one of")
    _replace_key(schema, "enum", "must be one of")
    _replace_key(schema, "allOf", "must be")
    _replace_key(schema, "minItems", "minimum length")
    _replace_key(schema, "maxItems", "maximum length")
    defs = schema.pop("$defs")
    required = schema.pop("required", [])
    _insert_ref(schema, defs)
    docs = _make_documentation(schema.pop("properties"), required=required)
    docs = docs.replace(
        "  **must be one of**:\n\n      **type**: string\n\n      **type**: null\n\n",
        "",
    )
    return docs


def _make_documentation(
    schema: list | dict | str,
    required: list[str] | None = None,
    level: int = 0,
    preface: str = "",
    element_seperator: str = "\n\n",
) -> str:
    indent = level * 2 * " "
    docs = []
    required = required if required is not None else []
    if isinstance(schema, dict):
        for key, val in schema.items():
            if key == "default" and not val:
                continue
            if key == "description":
                docs += [indent + val.replace("\n", " ")]
            elif key == "examples":
                normalized = textwrap.dedent(val)
                current_indent_level = indent + 2 * " "
                indented = textwrap.indent(
                    normalized,
                    current_indent_level,
                    lambda line: line.strip() != "",
                )
                if not indented.endswith(f"\n{current_indent_level}"):
                    indented = indented.rstrip() + f"\n{current_indent_level}"
                docs += [indent + f".. code-block:: yaml\n{indented}\n"]
            elif isinstance(val, dict):
                if key in required and level == 0:
                    key += "*"
                docs += [indent + f"**{key}**:"]
                docs += [_make_documentation(val, level=level + 1)]
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
                    element, level=level + 1, preface=" ", element_seperator="\n"
                )
            ]
        docs += list_docs
    else:
        schema = schema if isinstance(schema, str) else str(schema)
        docs += [indent + preface + schema]
    return element_seperator.join(docs)
