"""Where the records come from: the only module that knows the database schema.

Ids are upper-cased here. The tables disagree on case; MySQL compares ids case-insensitively but
Python joins do not, so one canonical form at the single entry point keeps every join downstream right.
"""
import contextlib

import polars as pl
import pymysql

from . import config


@contextlib.contextmanager
def connection():
    conn = pymysql.connect(charset='utf8mb4', **config.DB)
    try:
        yield conn
    finally:
        conn.close()


def query(sql, columns):
    """Rows of a query as a frame of strings, nulls as ''."""
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return pl.DataFrame(rows, schema=columns, orient='row').with_columns(pl.all().cast(pl.String).fill_null(''))


def addresses():
    """One permanent address per person. Present addresses are left out: in this data they are mostly
    hostel or campus, so they say nothing about where a family lives."""
    return query("SELECT UPPER(id), line1, line2, city, state, pincode FROM user_address WHERE type = 'permanent'",
                 ['id', 'line1', 'line2', 'city', 'state', 'pincode']).unique('id', keep='first', maintain_order=True)


def people():
    """Names, date of birth and parents. Used only after scoring (report, same-person check, evaluation),
    never as model input."""
    return query("""SELECT UPPER(d.id), TRIM(CONCAT_WS(' ', d.first_name, NULLIF(d.middle_name, ''), d.last_name)),
                           CAST(d.dob AS CHAR), d.last_name, o.father_name, o.mother_name
                    FROM user_details d LEFT JOIN user_other_details o ON o.id = d.id""",
                 ['id', 'name', 'dob', 'last', 'father', 'mother'])
