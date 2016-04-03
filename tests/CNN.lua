local CNN = require "../CNN"

-- TODO Check all backends
describe("CNN", function()
  describe("Model construction", function()
    it("Convolution module", function()
      local inputFrameSize  = 10  -- Sequence length
      local filterWidth     = 5
      local outputFrameSize = 3
      local module          = CNN.convolutionModule(inputFrameSize, outputFrameSize, filterWidth, "cpu")

      local frames  = 15  -- Batch size
      local classes = 25
      local tensor = torch.Tensor(frames, inputFrameSize):random(1, classes)

      -- Temporal convolution
      -- nOutputFrame x outputFrameSize
      -- nOutputFrame = (nInputFrame - kW) / dW + 1
      assert.are.same(
        module.modules[1]:forward(tensor):size():totable(),
        { (frames - filterWidth) / 1 + 1, outputFrameSize })

      -- Temporal convolution + max pooling
      local output = module:forward(tensor)
      assert.are.same(
        output:size():totable(),
        { outputFrameSize, outputFrameSize })
    end)
  end)
end)