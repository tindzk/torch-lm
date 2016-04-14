require 'busted.runner'()

local fun  = require "fun"
local Text = require "../Text"

describe("Text", function ()
  it("Vectorise tokens", function ()
    local tokens = fun.iter({ "a", "b", "ab" }):map(Text.tokeniseWord)
      local characters, characterToIndex = Text.lookupTable(tokens)
      assert.are.same(characters, {
        Symbols.ZeroPad,
        Symbols.WordStart,
        Symbols.WordEnd,
        "a",
        "b"
      })

    local tensor = Text.vectoriseTokens(tokens, characterToIndex)
    assert.are.same(tensor:totable(), {
      { 2, 4, 3, 1 },
      { 2, 5, 3, 1 },
      { 2, 4, 5, 3 }
    })
  end)

  it("Character tensor", function ()
    local characters, characterToIndex, tensor = Text.charsTensor("Hello")
    assert.are.same(characters, { [1] = "H", [2] = "e", [3] = "l", [4] = "o" })
    assert.are.same(characterToIndex, { ["H"] = 1, ["e"] = 2, ["l"] = 3, ["o"] = 4 })
    assert.are.same(tensor:totable(), { 1, 2, 3, 3, 4 })
  end)
end)
