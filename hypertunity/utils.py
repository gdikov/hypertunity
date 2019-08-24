# -*- coding: utf-8 -*-
import queue
import sys
import os
import importlib

GB_US_SPELLING = {"minimise": "minimize",
                  "maximise": "maximize",
                  "optimise": "optimize",
                  "emphasise": "emphasize"}

US_GB_SPELLING = {us: gb for gb, us in GB_US_SPELLING.items()}


def support_american_spelling(func):
    """Convert American spelling keyword arguments to British (default for hypertunity).

    Args:
        func: a Python callable to decorate.

    Returns:
        The decorated function which supports American keyword arguments.
    """
    def british_spelling_func(*args, **kwargs):
        gb_kwargs = {US_GB_SPELLING.get(kw, kw): val for kw, val in kwargs.items()}
        return func(*args, **gb_kwargs)

    return british_spelling_func


def join_strings(strings, join_char="_"):
    """Join list of strings with an underscore.

    The strings must contain string.printable characters only, otherwise an exception is raised.
    If one of the strings has already an underscore, it will be replace by a null character.

    Args:
        strings: iterable of strings.
        join_char: str, the character to join with.

    Returns:
        The joined string with an underscore character.

    Examples:
    ```python
        >>> join_strings(['asd', '', '_xcv__'])
        'asd__\x00xcv\x00\x00'
    ```

    Raises:
        ValueError if a string contains an unprintable character.
    """
    all_cleaned = []
    for s in strings:
        if not s.isprintable():
            raise ValueError("Encountered unexpected name containing non-printable characters.")
        all_cleaned.append(s.replace(join_char, "\0"))
    return join_char.join(all_cleaned)


def split_string(joined, split_char="_"):
    """Split joined string and substitute back the null characters with an underscore if necessary.

    Inverse function of `join_strings(strings)`.

    Args:
        joined: str, the joined representation of the substrings.
        split_char: str, the character to split by.

    Returns:
        A tuple of strings with the splitting character (underscore) removed.

    Examples:
    ```python
        >>> split_string('asd__\x00xcv\x00\x00')
        ('asd', '', '_xcv__')
    ```
    """
    strings = joined.split(split_char)
    strings_copy = []
    for s in strings:
        strings_copy.append(s.replace("\0", split_char))
    return tuple(strings_copy)


def drain_queue(q, close_queue=False):
    """Get all items from a queue until an `Empty` exception is raised.

    Args:
        q: `Queue`, the queue to drain.
        close_queue: bool, whether to close the queue, such that no other object can be put in. Default is False.

    Returns:
        List of all items from the queue.
    """
    items = []
    while True:
        try:
            it = q.get_nowait()
        except queue.Empty:
            break
        items.append(it)
    if close_queue:
        q.close()
    return items


def import_script(path):
    """Import a module or script by a given path.

    Args:
        path: str, can be either a module import of the form [package.]*[module]
            if the outer most package is in the PYTHONPATH, or a path to an arbitrary python script.

    Returns:
        The loaded python script as a module.
    """
    try:
        module = importlib.import_module(path)
    except ModuleNotFoundError:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find script {path}.")
        if not os.path.basename(path).endswith(".py"):
            raise ValueError(f"Expected a python script ending with *.py, found {os.path.basename(path)}.")
        import_path = os.path.dirname(os.path.abspath(path))
        sys.path.append(import_path)
        module = importlib.import_module(f"{os.path.basename(path).rstrip('.py')}",
                                         package=f"{os.path.basename(import_path)}")
        sys.path.pop()
    return module
