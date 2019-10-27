#
# Benchmark solvers and push results to pybamm-outputs GitHub
# https://gist.github.com/Maumagnaguagno/84a9807ed71d233e5d3f
#

# Clone outputs repository
git clone https://github.com/pybamm-team/pybamm-outputs.git
# Generate results
python3 results/benchmarking/time_models_and_solvers.py
mv -f results/benchmarking/README.md pybamm-outputs/benchmarking.md
# Set up git
cd pybamm-outputs
git remote
# git config --global user.email "travis@travis-ci.org"
# git config --global user.name "Travis CI"
# Current month and year, 2018-04-10
dateAndMonth=`date "+%F"`
# Stage the modified files in results 
git add benchmarking.md
# Create a new commit with a custom build message
git commit -m "Travis-generated results ($dateAndMonth)"
git remote rm origin
# Add new "origin" with access token in the git URL for authentication
git remote add origin https://${GITHUB_TOKEN}@github.com/pybamm-team/pybamm-outputs.git > /dev/null 2>&1
git push origin master --quiet
# Clean up
cd ..
rm -rf pybamm-outputs
