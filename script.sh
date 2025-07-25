#!bin/bash

set -e

(
    export PYTHONBREAKPOINT="pdbp.set_trace"
    python setup.py install
    (
        cd tests
        python -m pytest --tb=line -x
    )
)