-------------------------------------------------------------------------------
-- Format:
--  returns data
--  data . {train, test}
--  {train, test} . {inputs, labels}
--  inputs -> List of sequences
--  labels [task] -> List of labels
-------------------------------------------------------------------------------
require('pl.stringx').import()
local path = require('pl.path')
local utils = require('pl.utils')
local List = require('pl.List')
local seq = require('pl.seq')
require 'nn'

-- Parse cmd line
local opt = require('cmdlineargs')

assert(opt.task ~= "unknown", "Please input a task")
assert(opt.AAEmbedSize[1] > 0, "Need to use at least a unigram model for AAEmbedSize")

-- Returns a list of tensors for each line in the file
local function loadData(name, maxLoad)
    if maxLoad == 0 then maxLoad = 10000000000 end

    local data = seq.lines(name)
                    :take(maxLoad)
                    :map( function(s) return torch.Tensor(s:split(' ')) end)
                    :copy()

    --if maxLoad > 0 then data = tablex.sub(data, 0, maxLoad) end
    function data:size() return #data end

    return data
end


local jt = nn.JoinTable(1)

local function createDataset(size) 
    utils.printf("Making dataset...\n")
    local dir = opt.dataDir

    local aa = List.new()
    local psidata = List.new()

    print("Making AA...")
    for i=1,4 do if opt.AAEmbedSize[i] > 0 then 
        aa[i] = loadData( path.join(dir, string.format("aa%d.dat",i)) , size) 
    end end
    print("Making PSI...")
    for i=1,opt.PSINum do 
        print("Making psi "..i) 
        psidata[i] = loadData( path.join(dir, string.format("psi%d.dat.NR",i)) , size)
    end

    local labels = {}

    print("Making labels...")
    for task in opt.subtasks:iter() do
        print("Making label "..task)
        local curTag = task..".tag.dat"
        labels[task] = loadData( path.join(dir, curTag) , size)

        assert(labels[task]:size() == psidata[1]:size(),
            string.format("Length of outputs for task %s does not match length of inputs", task)
        )
        assert(labels[task]:size() == aa[1]:size(), 
            string.format("Length of embedding indicies for task %s does not match length of inputs", task)
        )
    end


    local data = aa:extend(psidata)
    local inputs = {}
    local outputs = {}
    setmetatable(inputs, {__index = function(self, ind)
        local ret = {}
        for i=1,data:len() do ret[i] = data[i][ind] end
        return ret
    end})
    setmetatable(outputs, {__index = function(self, ind)
        local ret = List:new()
        for _,task in ipairs(opt.subtasks) do
            ret:append(labels[task][ind])
        end
        return ret
    end})

    function inputs:size() return data[1]:size() end
    function outputs:size() return data[1]:size() end

    local dataset = {
        train={inputs={}, labels={}}, 
        valid={inputs={}, labels={}},
        test ={inputs={}, labels={}}
    }
    local function sizeSetter(set, size)
        function set.inputs:size() return size end
        function set.labels:size() return size end
    end

    if opt.CVfold == 0 then
        -- 3/5 train, 1/5 valid, 1/5 test
        local voffset = math.floor(3 * inputs:size() / 5)
        local toffset = math.floor(4 * inputs:size() / 5)

        local function indSetter(set, offset)
            setmetatable(set.inputs, {__index = function(self,i) return inputs[i+offset] end})
            setmetatable(set.labels, {__index = function(self,i) return outputs[i+offset] end})
        end

        indSetter(dataset.train, 0)
        indSetter(dataset.valid, voffset)
        indSetter(dataset.test , toffset)
        sizeSetter(dataset.train, voffset)
        sizeSetter(dataset.valid, toffset - voffset)
        sizeSetter(dataset.test , inputs:size()-toffset)
    else
        -- Use cross validation, no valid set
        local trindex = List.new()
        local teindex = List.new()
        for i=1,data[1]:size() do
            if (opt.CVfold-1)*inputs:size()/5 < i and i <= opt.CVfold*inputs:size()/5 then
                teindex:append(i)
            else trindex:append(i)
            end
        end
        setmetatable(dataset.train.inputs, {__index = function(self, i) return inputs[trindex[i]] end})
        setmetatable(dataset.train.labels, {__index = function(self, i) return outputs[trindex[i]] end})
        setmetatable(dataset.test.inputs, {__index = function(self, i) return inputs[teindex[i]] end})
        setmetatable(dataset.test.labels, {__index = function(self, i) return outputs[teindex[i]] end})
        local trsize = trindex:len()
        local tesize = teindex:len()
        sizeSetter(dataset.train, trsize)
        sizeSetter(dataset.valid, 0)
        sizeSetter(dataset.test , tesize)
    end

    local function sizeStats(s, x)
      print(x:size() .. " protein sequences in " .. s)
      local tot = 0
      for i=1,x:size() do
        tot = tot + x[i][1]:size(1)
      end
      print(tot .. " amino acids in " .. s)
    end

    sizeStats("train", dataset.train.labels)
    sizeStats("valid", dataset.valid.labels)
    sizeStats("test", dataset.test.labels)

    return dataset
end

local data = createDataset(opt.size)

return data
