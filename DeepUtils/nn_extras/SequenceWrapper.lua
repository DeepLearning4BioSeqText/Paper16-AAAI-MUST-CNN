local SequenceWrapper, parent = torch.class('nn.SequenceWrapper','nn.Module')

local function flatten(list)
    if type(list) ~= "table" then return {list} end
    local flat_list = {}
    for _, elem in ipairs(list) do
        for _, val in ipairs(flatten(elem)) do
            flat_list[#flat_list + 1] = val
        end
    end
    return flat_list
end

-- WARNING: If mlp contains params that are not called 'weight' and 'bias', this will probably fail
function SequenceWrapper:__init(mlp)
    parent.__init(self)
    self.mlp = mlp
    self.mlpfclones = {}
    self.ptbl = {}
    self.gptbl = {}
    self.mlpp, self.mlpgp = self.mlp:parameters()
    self.mlpp = flatten(self.mlpp)
    self.mlpgp = flatten(self.mlpgp)
    function self.mlpgp:zero()
        for _,v in ipairs(self) do v:zero() end
    end
    function self.mlpgp:add(const, other)
        for i,v in ipairs(self) do v:add(const, other[i]) end
    end
end

function SequenceWrapper:parameters()
    return self.mlpp, self.mlpgp
end

function SequenceWrapper:reset()
    self.mlp:reset()
    self.mlpfclones = {}
    self.ptbl = {}
    self.gptbl = {}
end

-- input is supposed to be a table of inputs, where each element is in the correct format
-- of the input of mlp
-- the outout is table of outputs
--
-- Input: {X}_{i=1}^{N}
-- Output: {Y}_i = mlp(X_i)
function SequenceWrapper:updateOutput(input)
    self.miter = 0
    self.output = {}
    --self.mlpfclones = {}

    -- run the set through mlp
    for i,example in ipairs(input) do
        local tmlp = self.mlpfclones[i]
        if not tmlp then
            tmlp = self.mlp:clone('weight', 'bias')
            local p, gp = tmlp:parameters()
            table.insert(self.mlpfclones, tmlp)
            table.insert(self.ptbl, flatten(p))
            table.insert(self.gptbl, flatten(gp))
        end
        local out = tmlp:updateOutput(example)
        table.insert(self.output, out)
        self.miter = self.miter + 1
    end
    return self.output
end

function SequenceWrapper:updateGradInput(input,gradOutput)
    self.gradInput = {}
    self.mlpgp:zero()
    for i,example in ipairs(input) do
        self.mlpfclones[i]:updateGradInput(example, gradOutput[i])
        table.insert(self.gradInput, self.mlpfclones[i].gradInput)
        self.mlpgp:add(1/self.miter, self.gptbl[i])
    end
    return self.gradInput
end

function SequenceWrapper:accGradParameters(input,gradOutput,scale)
    scale = scale or 1

    self.mlpgp:zero()
    for i,example in ipairs(input) do
        self.mlpfclones[i]:accGradParameters(example, gradOutput[i])
        self.mlpgp:add(1/self.miter, self.gptbl[i])
    end
end

function SequenceWrapper:__tostring__()
    local tab = '  '
    local line = '\n'
    local next = ' -> '
    local str = 'nn.SequenceWrapper'
    str = str .. ' {' .. tostring(self.mlp):gsub(line, line .. tab) .. ' }'
    return str
end
