--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Sumit Chopra <spchopra@fb.com>
--          Michael Mathieu <myrhev@fb.com>
--          Marc'Aurelio Ranzato <ranzato@fb.com>
--          Tomas Mikolov <tmikolov@fb.com>
--          Armand Joulin <ajoulin@fb.com>

-- This file contains a class RNNOption.
-- It parses the default options for RNNs and processes them.
-- Custom options can be added using option, optionChoice and
--   optionDisableIfNegative function.
-- The options are then parsed using the parse function.

require('os')
require('string')

local RNNOption = torch.class('RNNOption')

-- Init. Adds standard options.
function RNNOption:__init()
    self.cmd = torch.CmdLine()
    self.cmd.argseparator = '_'
    self.cmd:text()
    self.cmd:text('RNN Training for Language Modeling')
    self.cmd:text()
    self.cmd:text('Options:')

    self.options = {}

    -- dataset
    self:option('-dset',
                'dataset.name', 'europarl2015en',
                'Dataset name: Folder to contain train, valid and test text files')
    self:option('-threshold',
                'dataset.threshold', 0,
                'remove words appearing less than threshold')
    -- model
    self:option('-name',
                'model.name', 'srnn_sm',
                'name of the model: core_decoder. Cores: srnn | scrnn | lstm | gru | Decoders: sm | hsm | tsm')
    self:option('-nhid',
                'model.n_hidden', 512,
                'Number of hidden units')
    self:option('-nlayers',
                'model.n_layers', 2,
                'Number of recurrent layers')
    self:option('-dropout',
                'model.dropout', 0.2,
                'Dropout value between recurrent layers')
    self:optionChoice('-winit',
                      'model.w_init', 'frand',
                      'Weight matrix initialization (full, sparse, eye)',
                      {'frand', 'srand', 'eye'})
    self:option('-init',
                'model.initial_val', 0.05,
                'Value used for initialization of hidden units')
    self:option('-blen',
                'model.backprop_len', 35,
                'Number of steps to unfold back in time')
    
    -- trainer
    self:option('-batch_size',
                'trainer.batch_size', 64,
                'Size of mini-batch')
    self:option('-trbatches',
                'trainer.trbatches', -1,
                'Number of training batches. -1 = full data')
    self:option('-eta',
                'trainer.initial_learning_rate', 1,
                'Initial learning rate')
    self:option('-etashrink',
                'trainer.learning_rate_shrink', 2,
                'Learning rate shrink when validation error increases')
    self:option('-shrinkfactor',
                'trainer.shrink_factor', 0.9999,
                'multiplier on last validation error to decide on eta shrink')
    self:option('-shrinktype',
                'trainer.shrink_type', 'slow',
                'speed of learning rate annealing: at every epoch after the '
                    .. 'first anneal (fast) or after validation error '
                    .. 'stagnates (slow)')
    self:optionDisableIfNegative('-momentum',
                                 'trainer.momentum', 0,
                                 'Momentum (0 to disable)')
    self:option('-maxpt',
                'trainer.max_patience', 3,
                'Maximum number of iterations to wait when loss does not decrease')
    self:option('-gradclip',
				 'model.gradient_clip', 8,
				 'Norm of gradient clipping (0 to disable)')
    -- general
    self:option('-nepochs',
                'trainer.n_epochs', 50,
                'Number of training epochs')
    self:optionDisableIfNegative('-cuda', 'cuda_device', 1,
                                 'GPU device id (-1 for CPU)')
    self:option('-user',
                'user', '',
                'User. If none use uname',
                function (x)
                    if x == '' then return os.getenv('USER') else return x end
                end)
    self:option('-save',
                'trainer.save', false,
                'Whether to save the trained model or not')
    self:option('-load',
                'trainer.load', '',
                'Whether to load the trained model or not')
end

-- Adds an option:
--  cmd_option: the command line option (eg. -eta)
--  param_name: the name of the option in lua (the parse function returns a
--    table with all options. This is the index of the option in this table).
--    It be specialized to a subtable using a dot (eg. trainer.learning_rate)
--  default: the default value
--  process: a function to be applied to the parameter
function RNNOption:option(cmd_option, param_name, default, help, process_function)
    process_function = process_function or function(x) return x end
    self.cmd:option(cmd_option, default, help)
    local cmd_option_idx = cmd_option
    while cmd_option_idx:sub(1,1) == '-' do
        cmd_option_idx = cmd_option_idx:sub(2,-1)
    end
    self.options[param_name] = {cmd_option_idx, process_function}
end

-- Adds an option expecting a string. If the option is not in the list
-- <choices>, it raises an error.
function RNNOption:optionChoice(cmd_option, param_name, default, help, choices)
    local function f(x)
        for i = 1, #choices do
            if choices[i] == x then
                return x
            end
        end
        error('Option ' .. cmd_option .. ' cannot take value ' .. x
                  .. ' . Possible values are '
                  .. self:build_choices_string(choices))
    end
    self:option(cmd_option, param_name, default, help, f)
end

-- Adds an option expecting a number. It is replaced by nil if it is <= 0.
function RNNOption:optionDisableIfNegative(cmd_option, param_name, default, help)
    local function f(x)
        if x <= 0 then
            return nil
        else
            return x
        end
    end
    self:option(cmd_option, param_name, default, help, f)
end

-- Changes the default value to an option.
function RNNOption:change_default(cmd_option, new_default)
    if self.cmd.options[cmd_option] == nil then
        error('RNNOption: trying to change default, but option '
                  .. cmd_option .. ' does not exist')
    end
    self.cmd.options[cmd_option].default = new_default
end

function RNNOption:build_choices_string(choices)
    local out = '('
    for i = 1, #choices do
        if i ~= 1 then out = out .. '|' end
        out = out .. choices[i]
    end
    return out .. ')'
end

-- Parses the command line. It returns a table containing :
-- tables for the specialized options (eg. model, trainer, ...)
-- and the global parameters (eg. cuda_device)
function RNNOption:parse()
    local opt = self.cmd:parse(arg)
    local params = {}
    for k, v in pairs(self.options) do
        local cmd_option = v[1]
        local process_function = v[2]
        if k:find('.', 1, true) then
            local k1 = k:sub(1, k:find('.', 1, true)-1)
            local k2 = k:sub(k:find('.', 1, true)+1, -1)
            if params[k1] == nil then
                params[k1] = {}
            end
            params[k1][k2] = process_function(opt[cmd_option ])
        else
            params[k] = process_function(opt[cmd_option ])
        end
    end

    -- save dir
    local function to_string(x)
        if x == nil then
            return 'nil'
        elseif type(x) == 'boolean' then
            if x then
                return 'true'
            else
                return 'false'
            end
        else
            return x
        end
    end

    local mdir = "NCE_LSTM"
        .. '_bsz=' .. params.trainer.batch_size
        .. '_nhid=' .. params.model.n_hidden
        .. '_nlayers=' .. params.model.n_layers
        .. '_dropout=' .. params.model.dropout
        .. '_blen=' .. params.model.backprop_len
        .. '_lr=' .. params.trainer.initial_learning_rate
        .. '_lrs=' .. params.trainer.learning_rate_shrink
    if params.cuda_device ~= nil then
        mdir = mdir .. '_proc=gpu'
    else
        mdir = mdir .. '_proc=cpu'
    end
    local basedir = './output/'
     
    if params.trainer.save == true then
        params.trainer.save_dir = paths.concat(basedir, mdir)
    else
        params.trainer.save_dir = nil
    end

    -- extra
    params.dataset.seq_length = params.model.backprop_len
    params.dataset.batch_size = params.trainer.batch_size
    params.model.batch_size = params.trainer.batch_size
    params.model.cuda_device = params.cuda_device

    return params
end

-- prints the help
function RNNOption:text()
    self.cmd:text()
end

-- prints the value of the parameters <params>
function RNNOption:print_params(params)
    for k, v in pairs(params) do
        if type(v) == 'boolean' then
            if v then
                print('' .. k .. ': true')
            else
                print('' .. k .. ': false')
            end
        elseif type(v) ~= 'table' then
            print('' .. k .. ': ' .. v)
        end
    end
    for k, v in pairs(params) do
        if type(v) == 'table' then
            print('' .. k .. ':')
            for k2, v2 in pairs(v) do
                if type(v2) == 'boolean' then
                    if v2 then
                        print('  ' .. k2 .. ': true')
                    else
                        print('  ' .. k2 .. ': false')
                    end
                else
                    if type(v2) == 'table' then
                       print('  ' .. k2 .. ': table')
                    else
                       print('  ' .. k2 .. ': ' .. v2)
                    end
                end
            end
        end
    end
end
