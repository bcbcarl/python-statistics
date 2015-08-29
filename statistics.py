"""
Basic statistics module.

This module provides some functions introduced in Python 3.4.

Averages and measures of central location
-----------------------------------------

These functions calculate an average or typical value from a population or
sample.

==================  =============================================
Function            Description
==================  =============================================
mean()              Arithmetic mean ("average") of data.
median()            Median (middle value) of data.
mode()              Mode (most common value) of discrete data.
==================  =============================================

Measures of spread
------------------

These functions calculate a measure of how much the population or sample tends
to deviate from the typical or average values.

==================  =============================================
Function            Description
==================  =============================================
pstdev()            Population standard deviation of data.
pvariance()         Population variance of data.
stdev()             Sample standard deviation of data.
variance()          Sample variance of data.
==================  =============================================
"""

from __future__ import division, print_function, unicode_literals

import collections
import math

__all__ = [
    'StatisticsError',
    'mean',
    'median',
    'mode',
    'pstdev',
    'pvariance',
    'stdev',
    'variance',
]


class StatisticsError(ValueError):
    """StatisticsError is a subclass of ValueError."""
    pass


def mean(data):
    """Return the sample arithmetic mean of data.

    >>> mean([1, 2, 3, 4, 4])
    2.8

    >>> mean([-1.0, 2.5, 3.25, 5.75])
    2.625

    >>> from fractions import Fraction as F
    >>> mean([F(3, 7), F(1, 21), F(5, 3), F(1, 3)])
    Fraction(13, 21)

    >>> from decimal import Decimal as D
    >>> mean([D('0.5'), D('0.75'), D('0.625'), D('0.375')])
    Decimal('0.5625')

    If ``data`` is empty, StatisticsError will be raised.
    """
    if iter(data) is data:
        data = list(data)
    data_len = len(data)
    if data_len < 1:
        raise StatisticsError('mean requires at least one data point')
    return sum(data) / data_len


def median(data):
    """Return the median (middle value) of numeric data.

    >>> median([1, 3, 5])
    3

    >>> median([1, 3, 5, 7])
    4.0

    If ``data`` is empty, StatisticsError is raised.
    """
    data = sorted(data)
    data_len = len(data)
    if data_len == 0:
        raise StatisticsError('no median for empty data')
    if data_len % 2 == 1:
        return data[data_len // 2]
    if data_len % 2 == 0:
        i = data_len // 2
        return (data[i - 1] + data[i]) / 2


def _counts(data):
    """Return a count collection with the highest frequency.

    >>> _counts([1, 1, 2, 3, 3, 3, 3, 4])
    [(3, 4)]

    >>> _counts([2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5])
    [(1.25, 2)]
    """
    table = collections.Counter(iter(data)).most_common()
    if not table:
        return table
    maxfreq = table[0][1]
    for i in range(1, len(table)):
        if table[i][1] != maxfreq:
            table = table[:i]
            break
    return table


def mode(data):
    """Return the most common data point from discrete or nominal data.

    >>> mode([1, 1, 2, 3, 3, 3, 3, 4])
    3

    >>> mode([2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5])
    1.25

    If ``data`` is empty, or if there is not exactly one most common value,
    StatisticsError is raised.
    """
    data_len = len(data)
    if data_len == 0:
        raise StatisticsError('no mode for empty data')
    table = _counts(data)
    table_len = len(table)
    if table_len != 1:
        raise StatisticsError(
            'no unique mode; found %d equally common values' % table_len
        )
    return table[0][0]


def _ss(data, data_mean=None):
    """Return sum of square deviations of sequence data.

    >>> _ss([2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5])
    8.232142857142858

    >>> from decimal import Decimal as D
    >>> _ss([D('27.5'), D('30.25'), D('30.25'), D('34.5'), D('41.75')])
    Decimal('124.0750')

    >>> from fractions import Fraction as F
    >>> _ss([F(1, 6), F(1, 2), F(5, 3)])
    Fraction(67, 54)
    """
    if data_mean is None:
        data_mean = mean(data)
    return sum((x - data_mean)**2 for x in data)


def variance(data, xbar=None):
    """Return the sample variance of data.

    >>> data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
    >>> variance(data)
    1.3720238095238095

    >>> m = mean(data)
    >>> variance(data, m)
    1.3720238095238095

    >>> from decimal import Decimal as D
    >>> variance([D('27.5'), D('30.25'), D('30.25'), D('34.5'), D('41.75')])
    Decimal('31.01875')

    >>> from fractions import Fraction as F
    >>> variance([F(1, 6), F(1, 2), F(5, 3)])
    Fraction(67, 108)

    Raises StatisticsError if ``data`` has fewer than two values.
    """
    if iter(data) is data:
        data = list(data)
    data_len = len(data)
    if data_len < 2:
        raise StatisticsError('variance requires at least two data points')
    return _ss(data, xbar) / (data_len - 1)


def pvariance(data, mux=None):
    """Return the population variance of data.

    >>> data = [0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25]
    >>> pvariance(data)
    1.25

    >>> mu = mean(data)
    >>> pvariance(data, mu)
    1.25

    >>> from decimal import Decimal as D
    >>> pvariance([D('27.5'), D('30.25'), D('30.25'), D('34.5'), D('41.75')])
    Decimal('24.8150')

    >>> from fractions import Fraction as F
    >>> pvariance([F(1, 4), F(5, 4), F(1, 2)])
    Fraction(13, 72)

    Raises StatisticsError if ``data`` is empty.
    """
    if iter(data) is data:
        data = list(data)
    data_len = len(data)
    if data_len < 1:
        raise StatisticsError('pvariance requires at least one data point')
    return _ss(data, mux) / data_len


def stdev(data, xbar=None):
    """Return the square root of the sample variance.

    >>> stdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])
    1.0810874155219827
    """
    return math.sqrt(variance(data, xbar))


def pstdev(data, mux=None):
    """Return the square root of the population variance.

    >>> pstdev([1.5, 2.5, 2.5, 2.75, 3.25, 4.75])
    0.986893273527251
    """
    return math.sqrt(pvariance(data, mux))
