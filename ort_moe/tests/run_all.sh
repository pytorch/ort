#! /bin/bash
set -e
# Any subsequent(*) commands which fail will cause the shell script to exit immediately

echo "Running all tests"
echo "And print test coverage of the code files under folder of ../moe_module/ . If you don't need the coverage imformation, please replace \"coverage run --parallel-mode --source=../moe_module/\" with \"python\" to disable the coverage information collecting."

mpirun -n 4 --allow-run-as-root coverage run --parallel-mode --source=../moe_module/ -m pytest --with-mpi test_top2gating.py
mpirun -n 4 --allow-run-as-root coverage run --parallel-mode --source=../moe_module/ -m pytest --with-mpi test_moe.py
mpirun -n 4 --allow-run-as-root coverage run --parallel-mode --source=../moe_module/ -m pytest --with-mpi test_grid.py

echo "Combine the coverage tool output and print the report."
coverage combine
coverage report -m > coverage_log
python test_coverage.py

python test_uni_image.py

