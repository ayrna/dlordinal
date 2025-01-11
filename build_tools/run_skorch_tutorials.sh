#!/bin/bash

# Script to run all example notebooks.
set -euo pipefail

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"

included=(
    "tutorials/dlordinal_with_skorch_tutorial.ipynb"
)

shopt -s lastpipe
notebooks=()
runtimes=()

for notebook in "${included[@]}"; do
    echo "Running: $notebook"

    start=$(date +%s)
    $CMD "$notebook"
    end=$(date +%s)

    notebooks+=("$notebook")
    runtimes+=($((end-start)))
done

# print runtimes and notebooks
echo "Runtimes:"
paste <(printf "%s\n" "${runtimes[@]}") <(printf "%s\n" "${notebooks[@]}")
