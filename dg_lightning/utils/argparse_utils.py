
import datetime


def create_datetime_hash(fmt: str = "%Y-%m-%d_%H:%M:%S") -> str:
    return datetime.datetime.now().strftime(fmt)
