local fun     = require "fun"
local class   = require "class"
local Storage = require "Storage"
local Helpers = require "Helpers"
local Batch   = class("Batch")

-- Slice tensor to a multiple of `value`
function cutOffTensor(tensor, value)
  local length = tensor:size(1)

  if length % value ~= 0 then
    return tensor:sub(1, value * math.floor(length / value))
  end

  return tensor
end

-- Returns batches for `x` and `y`. Each batch has the dimension
-- `batchSize x sequenceLength x maximumTokenLength`.
function createBatches(tensor, maximumTokenLength, batchSize, sequenceLength)
  local xData = Helpers.init(tensor)
  local yData = Helpers.slice(tensor, 1)

  -- Cut off tensors so that we can use `batchSize` and `maximumLength` in view()
  local xDataCut = cutOffTensor(xData, batchSize * sequenceLength)
  local yDataCut = cutOffTensor(yData, batchSize * sequenceLength)

  -- Use view() to reshape the tensor in to a 3D matrix; -1 denotes the dimension
  -- to be inferred.
  local xBatches = xDataCut
    :view(batchSize, -1, maximumTokenLength)
    :split(sequenceLength, 2)  -- TODO Why 2?

  -- TODO Is this right?
  local yBatches = yDataCut
    :view(batchSize, -1, maximumTokenLength)
    :split(sequenceLength, 2)

  return {[1] = xBatches, [2] = yBatches}
end

-- `batchSize` is the number of sequences in a batch, trained in parallel
-- `sequenceLength` is the number of time steps of each sequence
function Batch:__init(dataDir, batchSize, sequenceLength)
  local vocabulary = torch.load(Storage.vocabularyFile(dataDir))
  local tensors    = torch.load(Storage.tensorFile(dataDir))

  self.symbols, self.symbolToIndex = table.unpack(vocabulary)

  self.trainingSet   = tensors[DataSet.TrainingSet]
  self.validationSet = tensors[DataSet.ValidationSet]
  self.testSet       = tensors[DataSet.TestSet]

  self.maximumTokenLength = self.trainingSet:size(2)

  print(string.format('Tokens count (training set): %d', self.trainingSet:size(1)))
  print(string.format('Tokens count (test set): %d', self.testSet:size(1)))
  print(string.format('Tokens count (validation set): %d', self.validationSet:size(1)))
  print(string.format('Token vocabulary size: %d', #self.symbols))
  print(string.format('Maximum token length: %d', self.maximumTokenLength))

  self.trainingBatches   = createBatches(self.trainingSet, self.maximumTokenLength, batchSize, sequenceLength)
  self.validationBatches = createBatches(self.validationSet, self.maximumTokenLength, batchSize, sequenceLength)
  self.testBatches       = createBatches(self.testSet, self.maximumTokenLength, batchSize, sequenceLength)

  print(string.format('Training batches: %d', #self.trainingBatches[1]))
  print(string.format('Validation batches: %d', #self.trainingBatches[1]))
  print(string.format('Test batches: %d', #self.testBatches[1]))
end

-- Returns all `x` and `y` batches for a DataSet split
function Batch:getBatches(split)
  if split == DataSet.TrainingSet then
    return self.trainingBatches[1], self.trainingBatches[2]
  elseif split == DataSet.TestSet then
    return self.testBatches[1], self.testBatches[2]
  elseif split == DataSet.ValidationSet then
    return self.validationBatches[1], self.validationBatches[2]
  else
    assert(false)
  end
end

function Batch:toText(tensor)
  return fun
    .iter(tensor:totable())
    :map(function (x) return self.symbols[x] end)
    :foldl(function (acc, cur) return acc .. cur end, "")
end

function Batch:sequenceToText(tensor)
  return fun
    .range(1, tensor:size(1))
    :map(function (x) return self.toText(self, tensor[x]) end)
    :foldl(function (acc, cur) return acc .. "\n" .. cur end, "")
end

return Batch