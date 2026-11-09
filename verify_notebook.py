"""Проверка сохранённого артефакта без повторного обучения моделей."""
from pathlib import Path
import nbformat


def main():
    path = Path(__file__).with_name('Lab3_Clustering.ipynb')
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    assert code, 'Нет исполняемых ячеек'
    assert all(c.execution_count is not None for c in code), 'Есть невыполненные ячейки'
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    assert not any('{{analysis:' in c.source for c in notebook.cells), 'Остались шаблоны'
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'cell_{i}', 'exec')
            assert notebook.cells[i + 1].cell_type == 'markdown', 'Нет интерпретации после кода'
    images = [o for c in code for o in c.outputs if 'image/png' in o.get('data', {})]
    assert images, 'Графики не сохранены'
    print(f'Valid notebook: {len(notebook.cells)} cells, {len(code)} executed code cells, '
          f'{len(images)} figures, 0 errors, 0 unresolved templates.')


if __name__ == '__main__':
    main()
