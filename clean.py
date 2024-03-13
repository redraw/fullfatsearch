import os
import sys
import sqlite_utils


EXCLUDE = [kw.strip() for kw in os.getenv("EXCLUDE", "").split(",") if kw]

if not EXCLUDE:
    print("nothing to do")
    sys.exit(0)

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
subs.optimize()
