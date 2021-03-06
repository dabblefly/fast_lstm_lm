
local LM = torch.class('nn.LanguageModel')

--~ TO DO: Add Noise Contrastive Estimation

function LM:__init(params, vocab_size)

	self.n_hiddens = params.n_hidden
	self.n_layers = params.n_layers
	self.dropout = params.dropout
	self.initial_val = params.initial_val
	self.vocab_size = vocab_size
	self.gradient_clip = params.gradient_clip
	
	--~ Model definition
	self.net = nn.Sequential()
	self.rnns = {}
	
	local V, H = self.vocab_size, self.n_hiddens
	self.net:add(nn.LookupTable(V, H))
	
	for i = 1, self.n_layers do
		
		self.net:add(nn.Dropout(self.dropout))
		
		
		local rnn = cudnn.LSTM(H, H, 1)
		table.insert(self.rnns, rnn)
		self.net:add(rnn)
		
		
		
	end
	
	
	self.net:add(nn.Dropout(self.dropout))	
	
	local linear = nn.Linear(H, V)
	local softmax = cudnn.LogSoftMax()
	
	self.net:add(nn.Sequencer(linear))
	self.net:add(nn.Sequencer(softmax))
	
	--~ Ship model to GPU
	--~ Does not support CPU due to using cudnn
	self.net:cuda()
	
	self.params, self.gradParams = self:getParameters()
	self.best_params = torch.CudaTensor(self.params:nElement()):copy(self.params)
	
	self.params:uniform(-self.initial_val, self.initial_val)
	
	local total_params = self.params:nElement()
	
	print("Total parameters of model: " .. total_params)
	
	local crit = nn.ClassNLLCriterion(nil, true):cuda()
	
	self.criterion = nn.SequencerCriterion(crit)
	
	print(self.net)
end

function LM:createHiddenInput(batch_size)
	
	for _, rnn in ipairs(self.rnns) do
		rnn.hiddenInput = torch.CudaTensor(1, batch_size, self.n_hiddens):fill(0)
		rnn.cellInput = torch.CudaTensor(1, batch_size, self.n_hiddens):fill(0)
	end
end

function LM:carryHiddenUnits()
	
	for _, rnn in ipairs(self.rnns) do
		rnn.hiddenInput:copy(rnn.hiddenOutput)
		rnn.cellInput:copy(rnn.cellOutput)
	end
	
	
end
	

function LM:eval(input, target)
	
	local N, T = input:size(1), input:size(2)
	local n_samples = N 
	
	
	local net_output = self.net:forward(input)
	
	self:carryHiddenUnits()
		
	local loss = self.criterion:forward(net_output, target)
	
	return loss, n_samples
	
	
	
end


function LM:trainBatch(input, target, learning_rate)
	
	self.net:zeroGradParameters()
	
	local N, T = input:size(1), input:size(2)
	
	local batch_size = T
	local n_samples = N 
	
	--~ Forward Pass	
	local net_output = self.net:forward(input)
	
	self:carryHiddenUnits()
	
	
	local loss = self.criterion:forward(net_output, target)
	
		
	local gradloss = self.criterion:backward(net_output, target)
	
	
	self.net:backward(input, gradloss)
	
	norm = self.gradParams:norm()

	if norm > self.gradient_clip then
		self.gradParams:mul(self.gradient_clip / norm)
	end	
	
	self.net:updateParameters(learning_rate)
	collectgarbage()
	
	return loss , n_samples
end


function LM:getParameters()
  return self.net:getParameters()
end

function LM:saveBestParams()
	
	self.best_params:copy(self.params)
end

function LM:revertBestParams()
	self.params:copy(self.best_params)
end

function LM:training()
  self.net:training()
end

function LM:evaluate()
  self.net:evaluate()
end

function LM:clearState()
  self.net:clearState()
end

function LM:resetStates()
	
	for _, rnn in ipairs(self.rnns) do
		rnn.hiddenInput:zero()
	end
end
