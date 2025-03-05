"""Read .sps file and convert to dataframe
For caveats please see:
https://dirkmjk.nl/en/2017/04/python-script-import-sps-files
"""

import re
import pandas as pd

PATTERN = r'\"(.*?)\"'


def get_cells(row):
    """Split row into cells"""
    cells = re.findall(PATTERN, row)
    return [c.strip() for c in cells]


def read_sps(path, encoding):
    """Read .sps file and convert to dataframe"""
    text = open(path, encoding=encoding).read()
    column_txt = text.split('VAR LABELS')[1].split('VALUE LABELS')[0]
    column_lines = column_txt.split('\n')
    columns = [line.split('"')[1] for line in column_lines if '"' in line]
    contains_data = False
    contains_labels = False
    new_row = None
    df = pd.DataFrame(columns=columns)
    labels = {}
    lines = text.split('\n')
    for line in lines:
        if line == 'END DATA.':
            contains_data = False
        if contains_data:
            cells = [cell for cell in re.split('\s+', line) if cell != '']
            if not new_row:
                new_row = cells
            else:
                new_row.extend(cells)
            if len(new_row) == len(columns):
                df.loc[len(df)] = new_row
                new_row = []
        if line == 'BEGIN DATA':
            contains_data = True
        if contains_labels:
            stripped = line.strip()
            if stripped.startswith('Key'):
                current = int(stripped[3:].split(' ')[0])
                label_columns = [columns[current], 'label_{}'.format(current)]
                labels[current] = pd.DataFrame(columns=label_columns)
            new_row = get_cells(line)
            if len(new_row) == 2:
                labels[current].loc[len(labels[current])] = new_row
        if line == 'VALUE LABELS':
            contains_labels = True
    for key in labels:
        df = df.merge(labels[key], on=columns[key], how='outer')
    return df
