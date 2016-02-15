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
