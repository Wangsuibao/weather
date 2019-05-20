import numpy
import logging
import theano
import theano.tensor as TT
from theano.gradient import grad_clip

from sparnn.utils import *
from sparnn.layers import Layer

logger = logging.getLogger(__name__)


class ConvLSTMLayer(Layer):
    def __init__(self, layer_param):
        super(ConvLSTMLayer, self).__init__(layer_param)
        if self.input is not None:
            assert 5 == self.input.ndim
        else:
            assert ("init_hidden_state" in layer_param or "init_cell_state" in layer_param)
        self.input_receptive_field = layer_param['input_receptive_field']
        self.transition_receptive_field = layer_param['transition_receptive_field']

        self.gate_activation = layer_param.get('gate_activation', 'sigmoid')
        self.modular_activation = layer_param.get('modular_activation', 'tanh')
        self.hidden_activation = layer_param.get('hidden_activation', 'tanh')

        self.init_hidden_state = layer_param.get("init_hidden_state", quick_theano_zero((self.minibatch_size,) + self.dim_out))
        self.init_cell_state = layer_param.get("init_cell_state", quick_theano_zero((self.minibatch_size,) + self.dim_out))
        self.init_hidden_state = TT.unbroadcast(self.init_hidden_state, *range(self.init_hidden_state.ndim))
        self.init_cell_state = TT.unbroadcast(self.init_cell_state, *range(self.init_cell_state.ndim))
        self.learn_padding = layer_param.get('learn_padding', False)
        self.input_padding = layer_param.get('input_padding', None)
        if self.input is None:
            assert 'n_steps' in layer_param
            self.n_steps = layer_param['n_steps']
        else:
            self.n_steps = layer_param.get('n_steps', self.input.shape[0])
        self.kernel_size = (self.feature_out, self.feature_in,
                            self.input_receptive_field[0], self.input_receptive_field[1])
        self.transition_mat_size = (self.feature_out, self.feature_out,
                                    self.transition_receptive_field[0], self.transition_receptive_field[1])

        #print('ConvLSTMLayer', self.kernel_size, self.transition_mat_size)
        self.W_hi = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_hi"))
        self.W_hf = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_hf"))
        self.W_ho = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_ho"))
        self.W_hc = quick_init_xavier(self.rng, self.transition_mat_size, self._s("W_hc"))

        if self.input is not None:
            self.W_xi = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xi"))
            self.W_xf = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xf"))
            self.W_xo = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xo"))
            self.W_xc = quick_init_xavier(self.rng, self.kernel_size, self._s("W_xc"))

        if self.learn_padding:
            self.hidden_padding = quick_zero((self.feature_out, ), self._s("hidden_padding"))
        else:
            self.hidden_padding = None

        self.b_i = quick_zero((self.feature_out, ), self._s("b_i"))
        self.b_f = quick_zero((self.feature_out, ), self._s("b_f"))
        self.b_o = quick_zero((self.feature_out, ), self._s("b_o"))
        self.b_c = quick_zero((self.feature_out, ), self._s("b_c"))

        self.W_ci = quick_zero((self.feature_out, ), self._s("W_ci"))
        self.W_cf = quick_zero((self.feature_out, ), self._s("W_cf"))
        self.W_co = quick_zero((self.feature_out, ), self._s("W_co"))
        if self.input is not None:
            self.param = [self.W_xi, self.W_hi, self.W_ci, self.b_i,
                          self.W_xf, self.W_hf, self.W_cf, self.b_f,
                          self.W_xo, self.W_ho, self.W_co, self.b_o,
                          self.W_xc, self.W_hc, self.b_c]
            if self.learn_padding:
                self.param.append(self.hidden_padding)
        else:
            self.param = [self.W_hi, self.W_ci, self.b_i,
                          self.W_hf, self.W_cf, self.b_f,
                          self.W_ho, self.W_co, self.b_o,
                          self.W_hc, self.b_c]
            if self.learn_padding:
                self.param.append(self.hidden_padding)
        self.is_recurrent = True
        self.fprop()
        #for i in self.param:
            #print('***********************gate',i.ndim)

    def set_name(self):
        self.name = "ConvLSTMLayer-" + str(self.id)

    def step_fprop(self, x_t, mask_t, h_tm1, c_tm1):
        print('*****************h_tm1 - c_tm1',h_tm1,c_tm1)
        #print('step fprop in conv lstm layer:', self.dim_in, self.kernel_size)
        if x_t is not None:
            # input_gate = x_t*W + h_t*W + c_t W
            input_gate = quick_activation(conv2d_same(x_t, self.W_xi, (None, ) + self.dim_in,
                                                      self.kernel_size, self.input_padding)
                                          + conv2d_same(h_tm1, self.W_hi, (None, ) + self.dim_out,
                                                        self.transition_mat_size, self.hidden_padding)
                                          + c_tm1 * self.W_ci.dimshuffle('x', 0, 'x', 'x')
                                          + self.b_i.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
            forget_gate = quick_activation(conv2d_same(x_t, self.W_xf, (None, ) + self.dim_in,
                                                       self.kernel_size, self.input_padding)
                                           + conv2d_same(h_tm1, self.W_hf, (None, ) + self.dim_out,
                                                         self.transition_mat_size, self.hidden_padding)
                                           + c_tm1 * self.W_cf.dimshuffle('x', 0, 'x', 'x')
                                           + self.b_f.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
            c_t = forget_gate * c_tm1 \
                  + input_gate * quick_activation(conv2d_same(x_t, self.W_xc, (None, ) + self.dim_in,
                                                              self.kernel_size, self.input_padding)
                                                  + conv2d_same(h_tm1, self.W_hc, (None, ) + self.dim_out,
                                                                self.transition_mat_size, self.hidden_padding)
                                                  + self.b_c.dimshuffle('x', 0, 'x', 'x'), "tanh")
            output_gate = quick_activation(conv2d_same(x_t, self.W_xo, (None, ) + self.dim_in,
                                                       self.kernel_size, self.input_padding)
                                           + conv2d_same(h_tm1, self.W_ho, (None, ) + self.dim_out,
                                                         self.transition_mat_size, self.hidden_padding)
                                           + c_t * self.W_co.dimshuffle('x', 0, 'x', 'x')
                                           + self.b_o.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
            h_t = output_gate * quick_activation(c_t, "tanh")

            print('gate*********',input_gate.ndim,forget_gate.ndim,output_gate.ndim,c_t.ndim,h_t.ndim)
        else:
            #input_gate = h_t * W
            input_gate = quick_activation(
                conv2d_same(h_tm1, self.W_hi, (None, ) + self.dim_out, self.transition_mat_size, self.hidden_padding)
                + c_tm1 * self.W_ci.dimshuffle('x', 0, 'x', 'x')
                + self.b_i.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
            forget_gate = quick_activation(conv2d_same(h_tm1, self.W_hf, (None, ) + self.dim_out,
                                                       self.transition_mat_size, self.hidden_padding)
                                           + c_tm1 * self.W_cf.dimshuffle('x', 0, 'x', 'x')
                                           + self.b_f.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
            c_t = forget_gate * c_tm1 \
                  + input_gate * quick_activation(conv2d_same(h_tm1, self.W_hc, (None, ) + self.dim_out,
                                                              self.transition_mat_size, self.hidden_padding)
                                                  + self.b_c.dimshuffle('x', 0, 'x', 'x'), "tanh")
            output_gate = quick_activation(conv2d_same(h_tm1, self.W_ho, (None, ) + self.dim_out,
                                                       self.transition_mat_size, self.hidden_padding)
                                           + c_t * self.W_co.dimshuffle('x', 0, 'x', 'x')
                                           + self.b_o.dimshuffle('x', 0, 'x', 'x'), "sigmoid")
            h_t = output_gate * quick_activation(c_t, "tanh")

        if mask_t is not None:
            h_t = mask_t * h_t + (1 - mask_t) * h_tm1
            c_t = mask_t * c_t + (1 - mask_t) * c_tm1

        #print h_t.ndim, c_t.ndim
        #h_t = quick_aggregate_pooling(h_t, "max", mask=None)
        #c_t = quick_aggregate_pooling(c_t, "max", mask=None)
        return h_t, c_t

    def init_states(self):
        return self.init_hidden_state, self.init_cell_state

    def fprop(self):

        # The dimension of self.mask is (Timestep, Minibatch).
        # We need to pad it to (Timestep, Minibatch, FeatureDim, Row, Col)
        # and keep the last three added dimensions broadcastable. TT.shape_padright
        # function is thus a good choice

        if self.mask is None:
            if self.input is not None:
                scan_input = [self.input]
                scan_fn = lambda x_t, h_tm1, c_tm1: self.step_fprop(x_t, None, h_tm1, c_tm1)
            else:
                scan_input = None
                scan_fn = lambda h_tm1, c_tm1: self.step_fprop(None, None, h_tm1, c_tm1)
        else:
            if self.input is not None:
                scan_input = [self.input, TT.shape_padright(self.mask, 3)]
                scan_fn = lambda x_t, mask_t, h_tm1, c_tm1: self.step_fprop(x_t, mask_t, h_tm1, c_tm1)
            else:
                scan_input = [TT.shape_padright(self.mask, 3)]
                scan_fn = lambda mask_t, h_tm1, c_tm1: self.step_fprop(None, mask_t, h_tm1, c_tm1)

        #print('conv lstm output:', scan_fn, self.init_cell_state, scan_input, self.n_steps)
        [self.output, self.cell_output], self.output_update = quick_scan(fn=scan_fn,
                                                                          outputs_info=[self.init_hidden_state,
                                                                                        self.init_cell_state],
                                                                          sequences=scan_input,
                                                                          name=self._s("lstm_output_func"),
                                                                          n_steps=self.n_steps
                                                                          )
        # no use print('**********************LSTM-out-shape',self.output.shape)
