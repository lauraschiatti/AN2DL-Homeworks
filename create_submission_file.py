#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
	create_submission_file.py: module for creating the formatted csv submission file.
"""

import os
from datetime import datetime


def create_csv(results, results_dir='./'):
    print("\nGenerating submission csv ... ")

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        fieldnames = 'Id,Category'
        f.write(fieldnames + '\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')
