#! /usr/bin/env python

from __future__ import print_function
import os
import traceback
import shutil
import nbformat
import nbconvert

def notebook_run(ipynb):
    with open(ipynb) as fh:
        nb = nbformat.reads(fh.read(), 4)

    exporter = nbconvert.PythonExporter()

    try:
        os.mkdir('figure')
    except:
        pass
    # source is a tuple of python source code
    # meta contains metadata
    source, meta = exporter.from_notebook_node(nb)
    try:
        exec(source.encode())
        shutil.rmtree('figure', ignore_errors=True)
        return []
    except:
        shutil.rmtree('figure', ignore_errors=True)
        return traceback.print_exc()


# @pytest.mark.parametrize("notebook", ipynbs)
def test_ipynb(notebook):
    error = notebook_run(notebook)
    assert error == []
