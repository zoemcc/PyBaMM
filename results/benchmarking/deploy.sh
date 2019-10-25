#
# Benchmark solvers and push results to GitHub
#

# Benchmark
python3 results/benchmarking/time_models_and_solvers.py

# Push
git checkout -b gh-pages
git add results/benchmarking/README.md
git commit -m "Travis CI: benchmark solvers"
git push --quiet --set-upstream origin gh-pages
