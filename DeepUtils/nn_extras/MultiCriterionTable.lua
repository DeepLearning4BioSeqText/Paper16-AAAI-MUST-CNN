local MultiCriterionTable, parent = torch.class('nn.MultiCriterionTable', 'nn.Criterion')

function MultiCriterionTable:__init()
   parent.__init(self)
   self.criterions = {}
   self.normalize = false
end

function MultiCriterionTable:add(criterion)
   table.insert(self.criterions,criterion)
end

function MultiCriterionTable:updateOutput(input, target)
   self.output = 0
   for i=1,#self.criterions do
      self.output = self.output + self.criterions[i]:forward(input[i],target[i])
   end
   if self.normalize then
      self.output = self.output / #self.criterions
   end
   return self.output
end

function MultiCriterionTable:updateGradInput(input, target)
   self.gradInput = {}
   for i=1,#self.criterions do
      if type(input[i]) == "table" then
         self.gradInput[i] = self.criterions[i]:backward(input[i], target[i])
      else
         self.gradInput[i] = torch.Tensor():resizeAs(input[i])
         self.gradInput[i]:zero()
         self.gradInput[i]:add(1, self.criterions[i]:backward(input[i], target[i]))

         if self.normalize then
            self.gradInput[i]:mul(1/#self.criterions)
         end
      end
   end
   return self.gradInput
end

function MultiCriterionTable:__tostring__()
    local tab = '  '
    local line = '\n'
    local next = '  |`-> '
    local ext = '  |    '
    local extlast = '       '
    local last = '   ... -> '
    local str = torch.type(self)
    str = str .. ' {' .. line .. tab .. 'input'
    for i=1,#self.criterions do
        if i == self.criterions then
            str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.criterions[i]):gsub(line, line .. tab .. extlast)
        else
            str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.criterions[i]):gsub(line, line .. tab .. ext)
        end
    end
    str = str .. line .. tab .. last .. 'output'
    str = str .. line .. '}'
    return str
end
