thread=$1
p=$2
VALDIR="../../data/validation/"
for entry in "0" "1" "2" "3" "4" "5"
do
  FILE=${VALDIR}simple_${entry}_1500.txt
  for method in "0" "1" "2" "3"
  do
    ./main -f $FILE -b $method -t 1
  done
done