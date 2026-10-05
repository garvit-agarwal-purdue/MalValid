# -*- coding: utf-8 -*-
#
# Vendored subset of TESSERACT's time-aware evaluation helpers
# (tesseract/temporal.py and tesseract/utils.py from
# https://github.com/s2labres/tesseract-ml-release, commit 05132c83eccaa0be24b5403bfe9a73c22199ddef).
# malvalid uses the installed `tesseract` package when it is importable and falls back to this
# copy otherwise (upstream pins numpy==1.22.4 and cannot be co-installed with modern numpy).
#
# Modifications for malvalid: only the helpers malvalid needs are kept (resolve_date,
# month_difference, get_relative_delta, time_aware_indexes); `utils.resolve_date` is inlined;
# code is otherwise unchanged.
#
# BSD 3-Clause License
#
# Copyright (c) 2018, Royal Holloway, University of London. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
#    conditions and the following disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
temporal.py
~~~~~~~~~~~

A module for working with and running time-aware evaluations. Most of the
functionality of this module falls into one of two categories: working with
arrays of datetimes or datetime-aligned series of data, and aggregating the
steps of the ML pipeline needed to conduct sound, time-aware evaluations.

"""
import bisect
import operator
from datetime import date, datetime

from dateutil.relativedelta import relativedelta


def resolve_date(d):
    """Convert a str or date to an appropriate datetime.

    Strings should be of the format '%Y', '%Y-%m or '%Y-%m-%d', for example:
    '2012', '1994-02' or '1991-12-11'. Date objects with no time information
    will be rounded down to the midnight beginning that date.

    Args:
        d (Union[str, date]): The string or date to convert.

    Returns:
        datetime: The parsed datetime equivalent of d.
    """
    if isinstance(d, datetime):
        return d

    if isinstance(d, date):
        return datetime.combine(d, datetime.min.time())

    for fmt in ('%Y', '%Y-%m', '%Y-%m-%d'):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            pass

    raise ValueError('date string format not recognized.')


def month_difference(d1, d2):
    """Get the difference in months between two datetimes."""
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def time_aware_indexes(t, train_size, test_size, granularity, start_date=None):
    """Return a list of indexes that partition the list t by time.

    Sorts the list of dates t before dividing into training and testing
    partitions, ensuring a 'history-aware' split in the ensuing classification
    task.


    Args:
        t (np.ndarray): Array of timestamp tags.
        train_size (int): The training window size W (in τ).
        test_size (int): The testing window size Δ (in τ).
        granularity (str): The unit of time τ, used to denote the window size.
            Acceptable values are 'year|quarter|month|week|day'.
        start_date (date): The date to begin partioning from (eg. to align with
            the start of the year).

    Returns:
        (list, list):
            Indexing for the training partition.
            List of indexings for the testing partitions.

    """
    # Order the dates as well as their original positions
    with_indexes = zip(t, range(len(t)))
    ordered = sorted(with_indexes, key=operator.itemgetter(0))

    # Split out the dates from the indexes
    dates = [tup[0] for tup in ordered]
    indexes = [tup[1] for tup in ordered]

    # Get earliest date
    start_date = resolve_date(start_date) if start_date else ordered[0][0]

    # Slice out training partition
    boundary = start_date + get_relative_delta(train_size, granularity)
    to_idx = bisect.bisect_left(dates, boundary)
    train = indexes[:to_idx]

    tests = []
    # Slice out testing partitions
    while to_idx < len(indexes):
        boundary += get_relative_delta(test_size, granularity)
        from_idx = to_idx
        to_idx = bisect.bisect_left(dates, boundary)
        tests.append(indexes[from_idx:to_idx])

    return train, tests


def get_relative_delta(offset, granularity):
    """Get delta of size 'granularity'.

    Args:
        offset: The number of time units to offset by.
        granularity: The unit of time to offset by, expects one of
            'year', 'quarter', 'month', 'week', 'day'.

    Returns:
        The timedelta equivalent to offset * granularity.

    """
    # Make allowances for year(s), quarter(s), month(s), week(s), day(s)
    granularity = granularity[:-1] if granularity[-1] == 's' else granularity
    try:
        return {
            'year': relativedelta(years=offset),
            'quarter': relativedelta(months=3 * offset),
            'month': relativedelta(months=offset),
            'week': relativedelta(weeks=offset),
            'day': relativedelta(days=offset),
        }[granularity]
    except KeyError:
        raise ValueError('granularity not recognised, try: '
                         'year|quarter|month|week|day')
