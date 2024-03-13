import os
import sys
import sqlite_utils


EXCLUDE = os.getenv("EXCLUDE", []).split(",")

db = sqlite_utils.Database(sys.argv[1])
subs = db["subtitles"]


def placeholders(n):
    return ",".join("?" for _ in range(n))


for keyword in EXCLUDE:
    ids = [i["rowid"] for i in subs.search(keyword)]
    where = f"rowid in ({placeholders(len(ids))})"

    with db.conn:
        subs.delete_where(where, ids, analyze=True)
        subs.rebuild_fts()
