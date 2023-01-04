from typing import *
import pickle as pkl

def convert_to_bytes(obj: Union[int, str, List[Union[str, int]]]) -> Union[bytes, List[bytes]]:
    """
    convert the input to bytes
    Args:
        obj (int or str or List[Union[str, int]]): the input to be converted

    Returns:
        the bytes type of the input

    Raises:
        ValueError: The number must be greater than zero
        TypeError: The data must be int or str or list[Union[str, int]]
    """
    if isinstance(obj, int):
        return convert_to_bytes(str(obj))
    if isinstance(obj, str):
        if str.isdigit(obj) and int(obj) < 0:
            raise ValueError(f'The number must be greater than zero, get {obj}')
        return obj.encode()
    elif isinstance(obj, list):
        return list(map(lambda x: convert_to_bytes(x), obj))
    else:
        raise TypeError(f"The data must be int or str or list[Union[str, int]], get {obj}")


def convert_to_str(obj: Union[bytes, List[bytes]]):
    """
    convert the input to string
    Args:
        obj (bytes or list[bytes]): the input to be converted

    Returns:
        the String type of the input

    Raises:
        TypeError: The data type must be bytes or list[bytes]
    """
    if isinstance(obj, bytes):
        if obj.startswith(b"\x80"):
            return pkl.loads(obj)
        else:
            return obj.decode()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return list(map(lambda x: convert_to_str(x), obj))
    else:
        raise TypeError(f"Error {obj}. The data type must be bytes or list[bytes]")
