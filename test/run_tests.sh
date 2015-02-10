DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
for file in ${DIR}/bin/*
do
  $file
done
for file in ${DIR}/python/*
do
  $file
done
echo "Done."
