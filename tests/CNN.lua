require 'busted.runner'()

local CNN = require "../CNN"

-- TODO Check all backends
describe("CNN", function()
  it("Convolution module", function()
    local inputFrameSize  = 10  -- Sequence length
    local filterWidth     = 5
    local outputFrameSize = 3
    local strideSize      = 1
    local module          = CNN.convolutionModule(inputFrameSize,
      outputFrameSize, filterWidth, strideSize, "cpu")

    local frames  = 15  -- Batch size
    local classes = 25
    local tensor = torch.Tensor(frames, inputFrameSize):random(1, classes)

    -- Temporal convolution
    -- nOutputFrame x outputFrameSize
    -- nOutputFrame = (nInputFrame - filterWidth) + 1
    assert.are.same(
      module.modules[1]:forward(tensor):size():totable(),
      { frames - filterWidth + 1, outputFrameSize })

    -- Temporal convolution + max pooling
    local output = module:forward(tensor)
    assert.are.same(
      output:size():totable(),
      { outputFrameSize, outputFrameSize })
  end)

  it("Narrow convolution", function()
    local batchSize   = 15
    local inputSize   = 17
    local alphabetLen = 25

    -- Dummy input similar to our real dataset
    local input = {
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen),
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen),
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen),
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen)
    }

    local charEmbeddingLen = 5
    local filterMinWidth   = 1
    local filterMaxWidth   = 10

    local net = CNN.narrowConvolution(inputSize, alphabetLen, charEmbeddingLen,
      filterMinWidth, filterMaxWidth, "cpu")
    local result = net:forward(input)

    assert.are.same(result[1]:size():totable(), {
      batchSize,
      torch.range(inputSize - (filterMaxWidth - filterMinWidth), inputSize):sum()
    })
  end)

  it("Padding", function()
    local batchSize   = 15
    local inputSize   = 17
    local alphabetLen = 25

    local input = {
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen),
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen),
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen),
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen)
    }

    local filterMinWidth = 1
    local filterMaxWidth = 10

    -- TODO Test functionality
    local padded = CNN.addPadding(input, filterMinWidth, filterMaxWidth)
  end)

  it("Wide convolution", function()
    local batchSize   = 15
    local inputSize   = 17
    local alphabetLen = 25

    -- Dummy input similar to our real dataset
    local input = {
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen),
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen),
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen),
      torch.Tensor(batchSize, inputSize):random(1, alphabetLen)
    }

    local charEmbeddingLen = 5
    local filterMinWidth   = 1
    local filterMaxWidth   = 10

    local padded = CNN.addPadding(input, filterMinWidth, filterMaxWidth)

    -- TODO Implement wide convolution
    -- local net = CNN.wideConvolution(inputSize, filterMinWidth, filterMaxWidth, "cpu")
    -- local result = net:forward(padded)
  end)
end)