-- Adds the following nn modules to the nn namespace
-- You can access them just by calling nn.UpDim
require 'nn'

include('nn_extras/UpDim.lua')
include('nn_extras/TemporalZip.lua')
include('nn_extras/MultiCriterionTable.lua')
include('nn_extras/SequenceWrapper.lua')
include('nn_extras/RandomMask.lua')
-- include('nn_extras/viterbi/init.lua')

package.path = package.path.. ';' .. paths.concat(paths.dirname(paths.thisfile()), "nn_extras/?.lua")
return {
    JointConfusionMatrix = require('JointConfusionMatrix'),
    Trainer = require('Trainer'),
}
