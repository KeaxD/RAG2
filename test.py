# smoke.py
import sys
print("Before anything:", sys.executable)
from unstructured.partition.auto import partition
print("After import partition")