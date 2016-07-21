local TemporalZip, parent = torch.class('nn.TemporalZip', 'nn.Module')

function TemporalZip:__init()
    parent.__init(self)
end

function TemporalZip:updateOutput(input)
    self.output = input:transpose(1,2):reshape(input:size(1)*input:size(2),input:size(3))
    return self.output
end

function TemporalZip:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:reshape(input:size(2), input:size(1), input:size(3)):transpose(1,2)
    return self.gradInput
end

local UpDim, parent = torch.class('nn.UpDim', 'nn.Module')

-- Turns a axbxc... tensor into a 1xaxbxc... tensor
UpDim.END = 0
UpDim.FRONT = 1

function UpDim:__init(position)
    parent.__init(self)
    if position == UpDim.FRONT then
        self.front = true
    elseif position == UpDim.END then
    self.front = false
else
    self.front = true
end
end

function UpDim:updateOutput(input)
    if self.front then
        self.output = input:reshape(1,unpack(input:size():totable()))
    else
        local dims = input:size():totable()
        table.insert(dims,1)
        self.output = input:reshape(unpack(dims))
    end
    return self.output
end

function UpDim:updateGradInput(input, gradOutput)
    if self.front then
        self.gradInput = gradOutput:select(1,1)
    else
        self.gradInput = gradOutput:select(input:dim()+1,1)
    end
    return self.gradInput
end

