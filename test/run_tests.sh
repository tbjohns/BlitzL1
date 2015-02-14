DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
for file in ${DIR}/bin/*
do
  if [[ ! -d $file && -x $file ]]
    then
    $file
  fi
done
for file in ${DIR}/python/*
do
  if [[ ! -d $file && -x $file ]]
    then
    python -W ignore $file
  fi
done
echo "Done."
