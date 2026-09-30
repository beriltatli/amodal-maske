"""Validate notebook schema/code, pinned source, checkpoint bytes and adapter scope."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

import nbformat
from download_weights import verify
from prepare_runtime import prepare, extract

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = nbformat.read(ROOT / 'notebooks/01_pcnet_m_colab.ipynb', as_version=4)
    nbformat.validate(notebook)
    code_cells = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'notebook-cell-{index}', 'exec')
            ast.parse(cell.source, feature_version=(3, 12))
            assert cell.execution_count is None and not cell.outputs
            code_cells += 1
            tree = ast.parse(cell.source)
            if tree.body and isinstance(tree.body[0], ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'BUNDLE' for target in tree.body[0].targets):
                for relative, content in ast.literal_eval(tree.body[0].value).items():
                    assert (ROOT / relative).read_text() == content, f'Stale notebook bundle: {relative}'
    for path in (ROOT / 'scripts').glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    repo, runtime = ROOT / 'third_party/deocclusion', ROOT / '.runtime/deocclusion'
    manifest = prepare(repo, runtime)
    # Only legacy NumPy aliases and explicit tensor placement may differ.
    original = (repo / 'inference.py').read_text()
    expected = re.sub(r'\bnp\.(int|bool|float)\b', r'\1', original)
    expected = expected.replace('.cuda()', '.to(next(model.model.parameters()).device)')
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse((runtime / 'inference.py').read_text()))
    for name, source in [('read_COCOA', 'datasets/reader.py'), ('expand_bbox', 'demos/demo_utils.py')]:
        assert ast.dump(ast.parse(extract((repo / source).read_text(), name))) == ast.dump(
            ast.parse(extract((runtime / 'official_helpers.py').read_text(), name)))
    weight_manifest = json.loads((ROOT / 'weights/manifest.json').read_text())
    digest = verify(ROOT / 'weights/COCOA_pcnet_m.pth.tar', weight_manifest)
    # A partial checkpoint must be rejected, not passed through to inference.
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as tmp:
        bad = Path(tmp) / 'partial.pth.tar'
        bad.write_bytes(b'<html>download failed</html>')
        try:
            verify(bad, weight_manifest)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid download accepted')
    packages = json.loads(subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'], text=True))
    result = {'notebook_schema': 'passed', 'code_cells_compiled': code_cells,
              'python312_syntax': 'passed', 'embedded_files_match_project': True,
              'upstream_commit': manifest['commit'], 'upstream_clean': True,
              'adapter_algorithm_check': 'passed', 'partial_download_rejected': True,
              'checkpoint_sha256': digest,
              'torch_installed': any(p['name'].lower() == 'torch' for p in packages),
              'note': 'Static checks only; inference evidence is stored in each output run.json'}
    (ROOT / 'reports/local-validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
