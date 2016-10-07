
pars=trainedModels/pretrainedRNNtheta.pik

declare -i cnt=0
for opt in sgd adam adagrad; 
do
  for kind in s1 s2;
  do
    for ph in T F;
    do 
      ((cnt++))
      exp=`printf "%03d" $cnt`
      odir=trainedModels/$exp
      mkdir $odir
      if  [ "$kind" == "s1" ]; then
        nohup python -u trainMath.py -exp $exp -o trainedModels/$exp -p $pars -k $kind -n 20 -l 0 -a 0 -opt $opt -ph $ph -nc F> trainedModels/$exp/$exp.out &
      else 	
	nohup python -u trainMath.py -exp $exp -o trainedModels/$exp -k $kind -n 20 -l 0 -a 0 -opt $opt -ph $ph -nc T > trainedModels/$exp/$exp.out & 	
      fi
    done;
  done;
done;

for exp in `ls trainedModels`;
do
  if [[ $exp =~ ^[0-9]+$ ]]; then 
    echo $exp; 
    ofile=trainedModels/$exp/$exp.out
    cat $ofile | grep 'MSPE:' | 
  fi
done;



declare -i cnt=0
for dh in 0 2 6; 
do   
  for ah in tanh relu; 
  do     
    e=`printf "%03d" $cnt`;     
    ((cnt++));
    nohup python -u keras_train.py -exp $e -o trainedModels/$e -n 200 -dh $dh -ah $ah > $e-$dh-$ah.out &    
  done; 
done;

