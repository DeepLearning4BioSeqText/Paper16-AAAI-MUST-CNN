aa="15,0,0,0"
tasks="dssp-mixed.2d"
nl=prelu
droparr=( 0 .2 .5)
indroparr=( 0 .1 .2)
name="exp6.19fixed"
mom=0.5

maxEpoch=40
model="conv"
param="9,7,3-2,2"
nhus="512,512,512"
nl="relu"
d=1


drop=${droparr[d]}
indrop=${indroparr[d]}


th ./main.lua -task $tasks -AAEmbedSize $aa -model $model -modelparams $param -nhus $nhus -nonlinearity $nl -maxEpochs $maxEpoch -dropout $drop -indropout $indrop -dataset 28Protein -momentum $mom -name $name
