require 'pl'
require 'nn'
require 'optim'

function writeData(fileString, dataset, name)
    header = nil
    traindata = {}
    validdata = {}
    testdata = {}
    for folder in io.popen(fileString):lines() do
        local p = path.join("../results", dataset, folder, "log.dat")
        if path.exists(p) then
            local log = torch.load(p)
            if not header then
                header = tablex.keys(log[1].matrix)
            end

            local best_model = tablex.imap(function(v)
                if not v then return 0 end
                return tablex.reduce('+', 
                    tablex.values(tablex.map(function(taskconf)
                            return taskconf.totalValid 
                        end,
                        v.validationmatrix
                    )
                )
            ) end, log)
            local _,bmi = torch.DoubleTensor(best_model):max(1)
            bmi = bmi[1]

            table.insert(traindata, tablex.map(function(v) return log[bmi].matrix[v].totalValid end, header))
            table.insert(traindata[#traindata], 1, folder)
            table.insert(traindata[#traindata], 2, bmi)
            local avgtime = torch.Tensor(tablex.imap(function(v) return v.time end, tablex.filter(log, function(v) return v ~= nil end))):mean()
            table.insert(traindata[#traindata], 3, avgtime/3600)

            local avgtime = torch.Tensor(tablex.imap(function(v) return v.validationtime end, tablex.filter(log, function(v) return v ~= nil end))):mean()
            table.insert(validdata, tablex.map(function(v) return log[bmi].validationmatrix[v].totalValid end, header))
            table.insert(validdata[#validdata], 1, folder)
            table.insert(validdata[#validdata], 2, bmi)
            table.insert(validdata[#validdata], 3, avgtime/3600)

            if log.test and log.test.matrix then
                table.insert(testdata, tablex.map(function(v) return log.test.matrix[v].totalValid end, header))
                table.insert(testdata[#testdata], 1, folder)
                table.insert(testdata[#testdata], 2, bmi)
                table.insert(testdata[#testdata], 3, log["test"].time/3600)
            end
        end
    end
    table.insert(header, 1, "model")
    table.insert(header, 2, "epoch")
    table.insert(header, 3, "time/epoch")

    function fmt(header, data)
        return  stringx.join("\t", header) .. '\n' .. 
                    stringx.join("\n", 
                        tablex.map(function(v) return stringx.join("\t", v) end, data)
                    )
    end

    file.write(name.."train.csv", fmt(header, traindata))
    file.write(name.."test.csv", fmt(header, testdata))
    file.write(name.."valid.csv", fmt(header, validdata))
end

writeData("ls ../results/28Protein/ | grep exp6.19", "28Protein", "28Protein")
-- writeData("ls ../results/4Protein/ | grep exp6.19", "4Protein", "4Protein")

