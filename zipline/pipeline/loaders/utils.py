import datetime

import pandas as pd
from pytz import utc


def normalize_data_query_time(dt, time, tz):
    """Apply the correct time and timezone to a date.

    Parameters
    ----------
    dt : datetime.datetime
        The original datetime that represents the date.
    time : datetime.time
        The time to query before.
    tz : tzinfo
        The timezone the time applies to.

    Returns
    -------
    query_dt : pd.Timestamp
        The timestamp with the correct time and date in utc.
    """
    # get the date after converting the timezone
    if dt.tzinfo is None:
        date = tz.localize(dt).date()
    else:
        date = dt.astimezone(tz).date()

    # merge the correct date with the time in the given timezone then convert
    # back to utc
    return pd.Timestamp(
        datetime.datetime.combine(date, time),
        tz=tz,
    ).tz_convert(utc)


def normalize_timestamp_to_query_time(df,
                                      time,
                                      tz,
                                      inplace=False,
                                      ts_field='timestamp'):
    """Update the timestamp field of a dataframe to normalize dates around
    some data query time/timezone.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to update. This needs a column named ``ts_field``.
    time : datetime.time
        The time to query before.
    tz : tzinfo
        The timezone the time applies to.
    inplace : bool, optional
        Update the dataframe in place.
    ts_field : str, optional
        The name of the timestamp field in ``df``.

    Returns
    -------
    df : pd.DataFrame
        The dataframe with the timestamp field normalized. If ``inplace`` is
        true, then this will be the same object as ``df`` otherwise this will
        be a copy.
    """
    dtidx = pd.DatetimeIndex(df[ts_field], tz='utc')
    # this mask represents the indicies where the time is greater than our
    # lookup time
    past_query_time_mask = dtidx.tz_convert(tz).time > time

    if not inplace:
        # don't mutate the dataframe in place
        df = df.copy()

    # for all of the times that are greater than our query time add 1
    # day and truncate to the date
    df.loc[past_query_time_mask, ts_field] = (
        dtidx[past_query_time_mask] + datetime.timedelta(days=1)
    ).normalize()
    # for all of the times that are less than our query time just truncate
    # to the date
    df.loc[~past_query_time_mask, ts_field] = pd.DatetimeIndex(
        df.loc[~past_query_time_mask, ts_field],
        tz='utc',
    ).normalize()
    return df
