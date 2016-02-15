-- This script allows you to test pre-trained models
-- Place your testing data into the "dataset" table and

require('cunn')
require('nn')
require('optim')
require('DeepUtils/nn_extras')

local opt = require('cmdlineargs')
print("Requiring data...") 
local dataset = require('data')

local model = torch.load(opt.modelparams)
print(model)

local set = {"train", "test", "valid"}
local confusion = op

local total = 0
local correct = 0
for _,s in pairs(set) do
    for i = 1, dataset[s].inputs:size() do
        local x = dataset[s].inputs[i]
        local y = dataset[s].labels[i][1]:cuda()
        local out = model:forward(x)[1]:narrow(1,1,y:size(1))
        local _,m = out:max(2)

        correct = correct + y:eq(m):sum()
        total = total + y:size(1)
    end
end

print(correct)
print(total)
print(correct/total)
