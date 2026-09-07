"""Run the real bootstrap in disposable processes, without starting the MCP server."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "ipython-mcp.py"

PROBE = """
import ast, json, os, sys
from pathlib import Path
from site import addsitedir

script, base, mode = sys.argv[1:]
base = Path(base)
own = base / 'own'
project = base / 'project'
project_site = project / '.venv' / ('Lib/site-packages' if sys.platform == 'win32' else 'lib/python3.13/site-packages')
os.chdir(project)
for key in ('PARENT', 'PROJECT_SITE_PACKAGES', 'VIRTUAL_ENV'):
    os.environ.pop(key, None)

if mode == 'parent':
    os.environ['PARENT'] = repr([str(own)])
    os.environ['PROJECT_SITE_PACKAGES'] = str(project_site)
else:
    addsitedir(str(own))
    sys.executable = str(own / 'bin/python')
    os.environ['VIRTUAL_ENV'] = str(own)

tree = ast.parse(Path(script).read_text())
# Stop at the runtime imports; execute all environment-selection code unchanged.
end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.ImportFrom) and node.module == 'asyncio')
exec(compile(ast.Module(body=tree.body[:end], type_ignores=[]), script, 'exec'))

import shared_probe, project_only_probe, own_editable_probe, project_editable_probe
print(json.dumps([shared_probe.VALUE, project_only_probe.VALUE, own_editable_probe.VALUE, project_editable_probe.VALUE]))
"""


class BootstrapTest(unittest.TestCase):
    def test_project_packages_are_fallbacks(self):
        for mode in ("same-version", "parent"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                own = base / "own"
                project_venv = base / "project/.venv"
                project_site = project_venv / ("Lib/site-packages" if sys.platform == "win32" else "lib/python3.13/site-packages")
                own_editable = base / "own-editable"
                project_editable = base / "project-editable"
                for directory in (own, project_site, own_editable, project_editable, own / "bin"):
                    directory.mkdir(parents=True, exist_ok=True)
                for venv in (own, project_venv):
                    (venv / "pyvenv.cfg").write_text("version = 3.13.0\n")
                python = project_venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
                python.parent.mkdir(exist_ok=True)
                python.touch()
                # Same-version discovery imports uv, but must not re-exec it.
                (own / "uv.py").write_text("def find_uv_bin():\n    raise AssertionError('unexpected re-exec')\n")
                for directory, modules in (
                    (own, {"shared_probe": "own"}),
                    (own_editable, {"own_editable_probe": "own editable"}),
                    (project_site, {"shared_probe": "project", "project_only_probe": "project only", "own_editable_probe": "wrong project override"}),
                    (project_editable, {"project_editable_probe": "project editable"}),
                ):
                    for name, value in modules.items():
                        (directory / f"{name}.py").write_text(f"VALUE = {value!r}\n")
                (own / "editable.pth").write_text(str(own_editable) + "\n")
                (project_site / "editable.pth").write_text(str(project_editable) + "\n")
                result = subprocess.run([sys.executable, "-I", "-S", "-c", PROBE, str(SCRIPT), temp, mode], capture_output=True, text=True, timeout=10, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), ["own", "project only", "own editable", "project editable"])


if __name__ == "__main__":
    unittest.main()
