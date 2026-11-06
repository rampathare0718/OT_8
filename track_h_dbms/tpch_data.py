"""
Track H: Synthetic TPC-H Schema & Database Generator
Populates SQLite with relational TPC-H tables:
- CUSTOMER (1,000 records)
- ORDERS (5,000 records)
- LINEITEM (15,000 records)
"""

import sqlite3
import numpy as np

def create_tpch_database(db_path=":memory:", seed=42):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA foreign_keys = OFF;")
    cur.execute("DROP TABLE IF EXISTS LINEITEM;")
    cur.execute("DROP TABLE IF EXISTS ORDERS;")
    cur.execute("DROP TABLE IF EXISTS CUSTOMER;")

    cur.execute("""
    CREATE TABLE CUSTOMER (
        c_custkey INTEGER PRIMARY KEY,
        c_name TEXT,
        c_mktsegment TEXT,
        c_acctbal REAL
    );
    """)

    cur.execute("""
    CREATE TABLE ORDERS (
        o_orderkey INTEGER PRIMARY KEY,
        o_custkey INTEGER,
        o_orderstatus TEXT,
        o_totalprice REAL,
        o_orderdate TEXT,
        o_orderpriority TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE LINEITEM (
        l_orderkey INTEGER,
        l_partkey INTEGER,
        l_suppkey INTEGER,
        l_linenumber INTEGER,
        l_quantity REAL,
        l_extendedprice REAL,
        l_discount REAL,
        l_shipdate TEXT,
        PRIMARY KEY (l_orderkey, l_linenumber)
    );
    """)

    rng = np.random.default_rng(seed)

    # 1. Populate CUSTOMER (1,000)
    n_cust = 1000
    segments = ["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD", "FURNITURE"]
    cust_data = []
    for c_id in range(1, n_cust + 1):
        seg = rng.choice(segments)
        bal = float(rng.uniform(-500, 10000))
        cust_data.append((c_id, f"Customer#{c_id:06d}", seg, bal))
    cur.executemany("INSERT INTO CUSTOMER VALUES (?, ?, ?, ?);", cust_data)

    # 2. Populate ORDERS (5,000)
    n_orders = 5000
    priorities = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
    statuses = ["O", "F", "P"]
    orders_data = []
    for o_id in range(1, n_orders + 1):
        c_fk = int(rng.integers(1, n_cust + 1))
        stat = rng.choice(statuses)
        price = float(rng.uniform(100, 50000))
        date = f"1995-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}"
        prio = rng.choice(priorities)
        orders_data.append((o_id, c_fk, stat, price, date, prio))
    cur.executemany("INSERT INTO ORDERS VALUES (?, ?, ?, ?, ?, ?);", orders_data)

    # 3. Populate LINEITEM (15,000)
    n_lineitems = 15000
    lineitem_data = []
    line_counts = {}
    for i in range(1, n_lineitems + 1):
        o_fk = int(rng.integers(1, n_orders + 1))
        line_num = line_counts.get(o_fk, 0) + 1
        line_counts[o_fk] = line_num
        qty = float(rng.integers(1, 50))
        ext_price = float(qty * rng.uniform(10, 500))
        disc = float(rng.uniform(0.0, 0.10))
        shipdate = f"1995-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}"
        lineitem_data.append((o_fk, int(rng.integers(1, 1000)), int(rng.integers(1, 100)), line_num, qty, ext_price, disc, shipdate))
    cur.executemany("INSERT INTO LINEITEM VALUES (?, ?, ?, ?, ?, ?, ?, ?);", lineitem_data)

    conn.commit()
    return conn

