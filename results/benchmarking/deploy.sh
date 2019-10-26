#
# Benchmark solvers and push results to GitHub
#

setup_git() {
  git config --global user.email "travis@travis-ci.org"
  git config --global user.name "Travis CI"
}

commit_benchmark_files() {
  git checkout master
  # Current month and year, e.g: Apr 2018
  dateAndMonth=`date "%F"`
  # Stage the modified files in results 
  git add results/benchmarking/README.md
  # Create a new commit with a custom build message
  # with "[skip ci]" to avoid a build loop
  git commit -m "Travis benchmarking: $dateAndMonth" -m "[skip ci]"
}

upload_files() {
  # Remove existing "origin"
  git remote rm origin
  # Add new "origin" with access token in the git URL for authentication
  git remote add origin https://${GITHUB_TOKEN}@github.com/pybamm-team/pybamm.git > /dev/null 2>&1
  git push origin master --quiet
}

# Benchmark
python3 results/benchmarking/time_models_and_solvers.py
setup_git
commit_benchmark_files
upload_files