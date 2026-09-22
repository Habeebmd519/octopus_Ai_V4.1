from pathlib import Path
import py_compile
py_compile.compile(str(Path(__file__).with_name("engine.py")), doraise=True)
print("Octapus V5 engine syntax: OK")
