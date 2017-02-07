require "rnn"
local fun = require "fun"

local CNN = {}

-- Module consisting of a temporal and a 1-max convolution
function CNN.convolutionModule(inputFrameSize, outputFrameSize, kernelWidth, strideSize, backend)
  local conv = nn.Sequential()
  if backend == "cl" then
    conv:add(nn.TemporalConvolution2(inputFrameSize, outputFrameSize, kernelWidth, strideSize))
    conv:add(nn.SpatialMaxPooling(outputFrameSize, 1))
  else
    conv:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kernelWidth, strideSize))
    conv:add(nn.TemporalMaxPooling(outputFrameSize))
  end
  return conv
end

-- FIXME: what the value for outputFrameSize should be is unclear for me
local outputFrameSize = 1
local strideSize      = 1

-- TODO Support regions (multiple filters with the same width)
-- http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
function CNN.narrowConvolution(inputSize, alphabetLen, charEmbeddingLen, filterMinWidth, filterMaxWidth, backend)
  local convModule = nn.ConcatTable()
  fun.range(filterMinWidth, filterMaxWidth):each(function (filterWidth)
    convModule:add(CNN.convolutionModule(charEmbeddingLen, outputFrameSize, filterWidth, strideSize, backend))
  end)

  local joinModule = nn.Sequential()
  joinModule:add(nn.JoinTable(1))
  local convOutputLength = torch.range(inputSize - (filterMaxWidth - filterMinWidth), inputSize):sum()
  joinModule:add(nn.Reshape(convOutputLength))

  local net = nn.Sequential()
  -- TODO Move the embedding layer out of this function?
  net:add(nn.LookupTable(alphabetLen, charEmbeddingLen))  -- Character embedding
  net:add(convModule)  -- Convolution
  net:add(joinModule)  -- Concatenate results into one vector

  return net
end

-- Wide convolutions honour borders by adding zero-padding
function CNN.wideConvolution(inputSize, filterMinWidth, filterMaxWidth, backend)
  local convModule = nn.ParallelTable()
  fun.range(filterMinWidth, filterMaxWidth):each(function (width)
    convModule:add(CNN.convolutionModule(inputSize + (width - 1) * 2, outputFrameSize, width, backend))
  end)

  local joinModule = nn.Sequential()
  joinModule:add(nn.JoinTable(1))
  -- TODO wrong calculations here
  local convOutputLength = torch.range(filterMinWidth, filterMaxWidth):sum()
  joinModule:add(nn.Reshape(convOutputLength))

  local net = nn.Sequential()
  -- TODO Add embedding layer for wide convolution
  net:add(convModule)
  net:add(joinModule)

  return net
end

-- Input must be padded when using wide convolutions
function CNN.addPadding(xTable, filterMinWidth, filterMaxWidth)
  return fun.iter(xTable):map(function (tensor)
    return fun.range(filterMinWidth, filterMaxWidth):map(function (width)
      if width == 1 then
        return tensor
      else
        local zeros = torch.zeros(tensor:size(1), width - 1)
        local res = torch.cat(zeros, tensor, 2)
        return torch.cat(res, zeros, 2)
      end
    end):totable()
  end):totable()
end

return CNN