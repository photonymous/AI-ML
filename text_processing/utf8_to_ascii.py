#!/usr/bin/env python3
import sys

from unidecode import unidecode

# read one line at a time from stdin. Data is in UTF-8 format, so we
# need to treat sys.stdin as UTF-8 encoding when we read it:
sys.stdin.reconfigure(encoding='utf-8')

# Sometimes there is an invalid UTF-8 character in the input stream.

while True:
    # Try to read a line from stdin:
    try:
        line = sys.stdin.readline()
    except UnicodeDecodeError:
        # If there is an invalid UTF-8 character, skip it and continue:
        # print the invalide line on stderr:
        sys.stderr.write("Invalid UTF-8 character in input stream:{}\n".format(line))

        continue
    # If we get an empty string, we have reached the end of the input:
    if not line:
        break
    # convert the line to ASCII:
    ascii_line = unidecode(line)
    # print the line to stdout:
    sys.stdout.write(ascii_line)



