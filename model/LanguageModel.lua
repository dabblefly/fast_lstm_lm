
local LM = torch.class('nn.LanguageModel')

--~ Added Noise Contrastive Estimation

function LM:__init(params, vocab_size, unigrams)

	self.n_hiddens = params.n_hidden
	self.n_layers = params.n_layers
	self.dropout = params.dropout
	self.initial_val = params.initial_val
	self.vocab_size = vocab_size
	self.gradient_clip = params.gradient_clip
	
	-- NCE parameters
	self.unigrams = unigrams
	self.Z = torch.exp(9)
	self.k = 10
	
	-- Model definition
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
	
	--~ Output of the LSTM is a tensor seq x batch x rnn_size
	
	self.net:add(nn.Dropout(self.dropout))	
	
	-- The split table splits the above tensor into a table of batch x rnn_size tensors 
	self.net:add(nn.SplitTable(1))
	
	-- NCE Module
	local nce = nn.NCEModule(H, V, self.k, self.unigrams, self.Z) 
	nce.weight:normal(0, 0.05)
	nce.bias:zero()
	nce.logsoftmax = true
	nce.normalized = true
	
	
	local lm = nn.Sequential()
		:add(nn.ParallelTable():add(self.net):add(nn.Identity()))
		:add(nn.ZipTable())
	
	-- We need the parallel table to send the target as well (needed for the NCE module)
	-- The ZipTable in the end does the following:
	
	-- (x1, x2, .... xn) --- (y1, y2, .... yn)
	-- transformed into
	-- (x1, y1), (x2, y2) ... (xn, yn)
		
	
	-- We feed the zipped couples to a sequencer - nce module
	lm:add(nn.Sequencer(nce))
	   
	self.targetmodule = nn.SplitTable(1):cuda()
	   	
	-- Sequence loss 
	local ncecrit = nn.NCECriterion():cuda() -- for training 
	local crit = nn.ClassNLLCriterion():cuda() -- for evaluation
	
	self.ncecrit = nn.SequencerCriterion(ncecrit)
	self.criterion = nn.SequencerCriterion(crit)
	
	
	-- Ship model to GPU
	-- Does not support CPU due to using cudnn
	self.net = lm
	self.net:cuda()
	
	self.params, self.gradParams = self:getParameters()
	self.params:uniform(-self.initial_val, self.initial_val)
	
	self.best_params = torch.CudaTensor(self.params:nElement()):copy(self.params)
	
	
	
	local total_params = self.params:nElement()
	
	print("Total parameters of model: " .. total_params)
	
	print(self.net)
end

-- Initializing the hidden states
function LM:createHiddenInput(batch_size)
	
	for _, rnn in ipairs(self.rnns) do
		rnn.hiddenInput = torch.CudaTensor(1, batch_size, self.n_hiddens):fill(0)
		rnn.cellInput = torch.CudaTensor(1, batch_size, self.n_hiddens):fill(0)
	end
end

-- Carry the hidden states to the next batch 
function LM:carryHiddenUnits()
	
	for _, rnn in ipairs(self.rnns) do
		rnn.hiddenInput:copy(rnn.hiddenOutput)
		rnn.cellInput:copy(rnn.cellOutput)
	end
	
	
end
	
-- Evaluation function
-- Note that the criterion used here is normal Negative Log-Likelihood
function LM:eval(input, target)
	
	local N, T = input:size(1), input:size(2)
	local n_samples = N
	
	target = self.targetmodule:forward(target)
	
	
	local net_output = self.net:forward({input, target})
	
	-- So that the next batch will continue from this batch 
	self:carryHiddenUnits()
	
	local loss = self.criterion:forward(net_output, target)
	
	return loss , n_samples
	
end


-- Training Function
-- Use Noise Constrastive Estimation / different loss module
function LM:trainBatch(input, target, learning_rate)
	
	self.net:zeroGradParameters()
	
	local N, T = input:size(1), input:size(2)
	
	local batch_size = T
	local n_samples = N 
	target = self.targetmodule:forward(target)
	
	--~ Forward Pass	
	local net_input = {input, target}
	local net_output = self.net:forward(net_input)
	
	self:carryHiddenUnits()
	
	
	local loss = self.ncecrit:forward(net_output, target)
	
	-- Backward Pass
	local gradloss = self.ncecrit:backward(net_output, target)
	
	
	self.net:backward(net_input, gradloss)
	
	
	-- Clip the gradient 
	local norm = self.gradParams:norm()

	if norm > self.gradient_clip then
		self.gradParams:mul(self.gradient_clip / norm)
	end	
	
	-- SGD Update
	-- This function allow each module to update accordingly
	-- The lookuptable will update sparsely which is faster than using the whole weight vector
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
