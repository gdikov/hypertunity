import queue

from functools import wraps

GB_US_SPELLING = {
    "minimise": "minimize",
    "maximise": "maximize",
    "optimise": "optimize",
    "optimiser": "optimizer",
    "emphasise": "emphasize"
}

US_GB_SPELLING = {us: gb for gb, us in GB_US_SPELLING.items()}


def support_american_spelling(func):
    """Convert American spelling keyword arguments to British (default for hypertunity).

    Args:
        func: a Python callable to decorate.

    Returns:
        The decorated function which supports American keyword arguments.
    """

    # using functools.wraps(func) enables automated documentation generation
    # for more information see: https://github.com/sphinx-doc/sphinx/issues/3783
    @wraps(func)
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
