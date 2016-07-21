require('pl.stringx').import()
local path = require('pl.path')
local utils = require('pl.utils')
local List = require('pl.List')
local seq = require('pl.seq')
require 'nn'

local function loaddata(name)
    local data = seq.lines(name)
                    :take(10000000000)
                    :map( function(s) return torch.Tensor(s:split(' ')) end)
                    :copy()

    function data:size() return #data end

    return data
end

datadir = params.datadir
aafile = "aa1.dat"
trainlabelfile = params.task..'.tag.dat'
print(trainlabelfile)

local function load()
	utils.printf("Start loading data\n")
	local aa = loaddata( path.join(datadir, aafile))
	local labels = loaddata(path.join(datadir,trainlabelfile))

    local dataset = {
        train={inputs={}, labels={}},
        valid={inputs={}, labels={}},
        test ={inputs={}, labels={}}
    }

    local function sizeSetter(set, size)
        function set.inputs:size() return size end
        function set.labels:size() return size end
    end

    local voffset = math.floor(3 * aa:size() / 5)
    local toffset = math.floor(4 * aa:size() / 5)

    local function indSetter(set, offset)
        setmetatable(set.inputs, {__index = function(self,i) return aa[i+offset] end})
        setmetatable(set.labels, {__index = function(self,i) return labels[i+offset] end})
    end

    indSetter(dataset.train, 0)
    indSetter(dataset.valid, voffset)
    indSetter(dataset.test , toffset)
    sizeSetter(dataset.train, voffset)
    sizeSetter(dataset.valid, toffset - voffset)
    sizeSetter(dataset.test , aa:size() - toffset)
    
    return dataset
end

return load()