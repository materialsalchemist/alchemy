import kuzu
import os

KUZU_DB_PATH = os.environ.get("KUZU_DB_PATH", "reaction_space_results/kuzu_db")
db = None
conn = None


def get_conn():
	global db, conn
	if conn is not None:
		return conn

	if not os.path.exists(KUZU_DB_PATH):
		return None

	try:
		# Cap buffer pool at 4GB to prevent system-wide memory mapping issues
		# and ensure stability with other C++ extensions.
		db = kuzu.Database(KUZU_DB_PATH, buffer_pool_size=4 * 1024 * 1024 * 1024)
		conn = kuzu.Connection(db)
		print(f"Connected to KuzuDB at {KUZU_DB_PATH} (4GB Pool)")
		return conn
	except Exception as e:
		print(f"Error connecting to KuzuDB: {e}")
		return None
