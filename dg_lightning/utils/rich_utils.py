
import typing
import argparse

from rich.console import Console
from rich.table import Table

def print_args_as_table(args: argparse.Namespace, title: str = 'Configuration'):
    
    console = Console(color_system='256')
    table = Table(title=title)
    table.add_column('Name', justify='right', style='white')
    table.add_column('Value', justify='left', style='green')
    _ = [table.add_row(str(k), str(v)) for k, v in vars(args).items()]
    console.print(table)
