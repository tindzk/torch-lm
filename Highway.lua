-- source: https://github.com/yoonkim/lstm-char-cnn

require "nngraph"

local HighwayMLP = {}

function HighwayMLP.mlp(size, numLayers, bias, f)
    -- size = dimensionality of inputs
    -- numLayers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)

    local output, transformGate, carryGate
    local numLayers = numLayers or 1
    local bias = bias or -2
    local f = f or nn.ReLU()
    local input = nn.Identity()()
    local inputs = {[1]=input}
    for i = 1, numLayers do
        output = f(nn.Linear(size, size)(inputs[i]))
        transformGate = nn.Sigmoid()(nn.AddConstant(bias)(nn.Linear(size, size)(inputs[i])))
        carryGate = nn.AddConstant(1)(nn.MulConstant(-1)(transformGate))
        output = nn.CAddTable()({
            nn.CMulTable()({transformGate, output}),
            nn.CMulTable()({carryGate, inputs[i]})	})
        table.insert(inputs, output)
    end
    return nn.gModule({input},{output})
end

return HighwayMLP