description = """
Performs inplace string replacement in a file using sed.

ARGS
FROM: regex to match what to replace
TO: expression to replace the match
FILE: name of file to perform replacement in.

Example:
FORWARD_MODEL_JOB REPLACE_STRING(<FROM>=something, <TO>=else, <FILE>=file.txt)
 > replace all something to else in file.txt

"""

category = "utility.templating"
