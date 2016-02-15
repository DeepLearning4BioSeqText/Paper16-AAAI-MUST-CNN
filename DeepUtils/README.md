# DeepUtils
A collection of utility packages that are project independent.

Usage
===
Clone this into your project repository, and use the files by something akin to 

```
require "DeepUtils/nn_extras"
```

If you are making new packages, please allow them to be included like that so we keep a consistent format.

##nn\_extras

###nn\_extras.JointConfusionMatrix
Gives many confusion matricies, for each task for joint problems. Initialize with nn\_extras.JointConfusionMatrix(tasklist, nClasses). Then, operation is exactly the same as nn.ConfusionMatrix, except input is a list of inputs to a regular confusion matrix, one for each class in the order of tasklist.

###nn\_extras.Trainer
Wraps a generic torch7-style trainer for multilayered networks using any number of optimization schemes from the `optim` package. Call either train or test with (model, criterion, x, y, params) to train or test the model. 

Params to the and defaults, * indicates params only applicable to train:
```
*optimMethod(optim.sgd): Optimizer to use
*batchSize(1): Minibatch size
progressbar(true): Whether to display a progress bar or not for training
doAfterEveryExample(function(output, label, err) {}): Function to run after each example is run.
*optimState({}): State of the optimizer. Passed in as the config to optimizers in optim chosen.
```

###nn.MultiCriterionTable
Insert many criterion. Layer accepts input of tables, and outputs the average error. Similar to the ParallelTable, for criterions.

```
+----------+         +-----------+
| {input1, +---------> {member1, |
|          |         |           |
|  input2, +--------->  member2, +------> error
|          |         |           |
|  input3} +--------->  member3} |
+----------+         +-----------+
```

###nn.UpDim
Takes in arbitrary input with dim `D = d1 x d2 x ... x dn`.
Argument can be nn.UpDim.END or nn.UpDim.FRONT. If it is a END UpDim layer, output will be D x 1. Otherwise, output is 1 x D.

###nn.TemporalZip
Input is a `B x T x m` tensor. "Zips" the tensor along the first dimension, and turns it into a `B*T x m` tensor, where `input[b, t, m] = output[t*B+b, m]`

###nn.SequenceWrapper
Takes in a network as input. Careful that network parameters are only `weight` and `bias`, otherwise this layer will not work.
input is supposed to be a table of inputs, where each element is in the correct format
of the input of mlp
the outout is table of outputs

```
Input: X_{i=1}^{N}
Output: Y_i = mlp(X_i)
```

###nn.Viterbi
Implements the sentence-level log likelihood layer in NLP from Scratch http://arxiv.org/pdf/1103.0398.pdf

## ProFi

Packaged lua profiler from https://gist.github.com/perky/2838755.

To use, do something like 

```
local ProFi = require('DeepUtils/ProFi')
ProFi:start()
main()
ProFi:stop()
ProFi:writeReport('profile.log')
```

##SLURM.lua

SLURM.SLURMBatchRun takes in a function that takes n arguments, and n tables of arguments. The function is to
output a slurm batch script to run, as a string. It passes every combination of the n tables of arguments to the
function, and then runs it using sbatch.
Here is an informative example on how to use this:

```
local slurm = require('DeepUtil/SLURM')

local script = [[
#!/bin/bash
#SBATCH -p qdata
#SBATCH -o protein28tasks.%s,%s.out

/usr/cs/bin/th ./main.lua -task %s -AAEmbedSize %s -model %s -modelparams %s -nhus %s -nonlinearity %s -maxEpochs %d -dropout %s -indropout %s -dataset 28Protein -noprogressbar]]
local task = "dssp-mixed.2d#stride-mixed.2d"
local aa = "15,0,0,0"

local function formattermlp (nhu, nonlinearity, dropout)
    local model = "mlp"
    local dp = dropout and ".5" or "0"
    local idp = dropout and ".2" or "0"
    return string.format(script, model, nhu, task, aa, model, "13", nhu, nonlinearity, 20, dp, idp)
end
print(formattermlp("128,128", "relu", true))

slurm.SLURMBatchRun(formattermlp, {"128,128", "64,64", "64,64,64"}, {"tanh", "relu", "prelu"}, {true, false})
```
