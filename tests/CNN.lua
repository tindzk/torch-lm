local CNN = require "../CNN"

-- TODO Check all backends
describe("CNN", function()
  describe("Model construction", function()
    it("Convolution module", function()
      local inputFrameSize  = 10  -- Sequence length
      local filterWidth     = 2
      local outputFrameSize = 1
      local module          = CNN.convolutionModule(inputFrameSize, outputFrameSize, filterWidth, "cpu")

      local frames  = 15  -- Batch size
      local classes = 25
      local tensor = torch.Tensor(frames, inputFrameSize):random(1, classes)
      local output = module:forward(tensor)
      assert.are.same(output:size():totable(), { 14, 1 })
    end)
  end)
end)