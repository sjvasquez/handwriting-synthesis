from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import _concat, _like_rnncell
from tensorflow.python.ops.rnn import _maybe_tensor_shape_from_tensor
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape
from tensorflow.python.eager import context


def raw_rnn(cell, loop_fn, parallel_iterations=None, swap_memory=False, scope=None):
    """
    raw_rnn adapted from the original tensorflow implementation
    (https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn.py)
    to emit arbitrarily nested states for each time step (concatenated along the time axis)
    in addition to the outputs at each timestep and the final state

    returns (
        states for all timesteps,
        outputs for all timesteps,
        final cell state,
    )
    """
    if not _like_rnncell(cell):
        raise TypeError("cell must be an instance of RNNCell")
    if not callable(loop_fn):
        raise TypeError("loop_fn must be a callable")

    parallel_iterations = parallel_iterations or 32

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "rnn") as varscope:
        if context.in_graph_mode():
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)

        time = constant_op.constant(0, dtype=dtypes.int32)
        (elements_finished, next_input, initial_state, emit_structure,
         init_loop_state) = loop_fn(time, None, None, None)
        flat_input = nest.flatten(next_input)

        # Need a surrogate loop state for the while_loop if none is available.
        loop_state = (init_loop_state if init_loop_state is not None
                      else constant_op.constant(0, dtype=dtypes.int32))

        input_shape = [input_.get_shape() for input_ in flat_input]
        static_batch_size = input_shape[0][0]

        for input_shape_i in input_shape:
            # Static verification that batch sizes all match
            static_batch_size.merge_with(input_shape_i[0])

        batch_size = static_batch_size.value
        const_batch_size = batch_size
        if batch_size is None:
            batch_size = array_ops.shape(flat_input[0])[0]

        nest.assert_same_structure(initial_state, cell.state_size)
        state = initial_state
        flat_state = nest.flatten(state)
        flat_state = [ops.convert_to_tensor(s) for s in flat_state]
        state = nest.pack_sequence_as(structure=state,
                                      flat_sequence=flat_state)

        if emit_structure is not None:
            flat_emit_structure = nest.flatten(emit_structure)
            flat_emit_size = [emit.shape if emit.shape.is_fully_defined() else
                              array_ops.shape(emit) for emit in flat_emit_structure]
            flat_emit_dtypes = [emit.dtype for emit in flat_emit_structure]
        else:
            emit_structure = cell.output_size
            flat_emit_size = nest.flatten(emit_structure)
            flat_emit_dtypes = [flat_state[0].dtype] * len(flat_emit_size)

        flat_state_size = [s.shape if s.shape.is_fully_defined() else
                           array_ops.shape(s) for s in flat_state]
        flat_state_dtypes = [s.dtype for s in flat_state]

        flat_emit_ta = [
            tensor_array_ops.TensorArray(
                dtype=dtype_i,
                dynamic_size=True,
                element_shape=(tensor_shape.TensorShape([const_batch_size])
                               .concatenate(_maybe_tensor_shape_from_tensor(size_i))),
                size=0,
                name="rnn_output_%d" % i
            )
            for i, (dtype_i, size_i) in enumerate(zip(flat_emit_dtypes, flat_emit_size))
        ]
        emit_ta = nest.pack_sequence_as(structure=emit_structure, flat_sequence=flat_emit_ta)
        flat_zero_emit = [
            array_ops.zeros(_concat(batch_size, size_i), dtype_i)
            for size_i, dtype_i in zip(flat_emit_size, flat_emit_dtypes)]

        zero_emit = nest.pack_sequence_as(structure=emit_structure, flat_sequence=flat_zero_emit)

        flat_state_ta = [
            tensor_array_ops.TensorArray(
                dtype=dtype_i,
                dynamic_size=True,
                element_shape=(tensor_shape.TensorShape([const_batch_size])
                               .concatenate(_maybe_tensor_shape_from_tensor(size_i))),
                size=0,
                name="rnn_state_%d" % i
            )
            for i, (dtype_i, size_i) in enumerate(zip(flat_state_dtypes, flat_state_size))
        ]
        state_ta = nest.pack_sequence_as(structure=state, flat_sequence=flat_state_ta)

        def condition(unused_time, elements_finished, *_):
            return math_ops.logical_not(math_ops.reduce_all(elements_finished))

        def body(time, elements_finished, current_input, state_ta, emit_ta, state, loop_state):
            (next_output, cell_state) = cell(current_input, state)

            nest.assert_same_structure(state, cell_state)
            nest.assert_same_structure(cell.output_size, next_output)

            next_time = time + 1
            (next_finished, next_input, next_state, emit_output,
             next_loop_state) = loop_fn(next_time, next_output, cell_state, loop_state)

            nest.assert_same_structure(state, next_state)
            nest.assert_same_structure(current_input, next_input)
            nest.assert_same_structure(emit_ta, emit_output)

            # If loop_fn returns None for next_loop_state, just reuse the previous one.
            loop_state = loop_state if next_loop_state is None else next_loop_state

            def _copy_some_through(current, candidate):
                """Copy some tensors through via array_ops.where."""
                def copy_fn(cur_i, cand_i):
                    # TensorArray and scalar get passed through.
                    if isinstance(cur_i, tensor_array_ops.TensorArray):
                        return cand_i
                    if cur_i.shape.ndims == 0:
                        return cand_i
                    # Otherwise propagate the old or the new value.
                    with ops.colocate_with(cand_i):
                        return array_ops.where(elements_finished, cur_i, cand_i)
                return nest.map_structure(copy_fn, current, candidate)

            emit_output = _copy_some_through(zero_emit, emit_output)
            next_state = _copy_some_through(state, next_state)

            emit_ta = nest.map_structure(lambda ta, emit: ta.write(time, emit), emit_ta, emit_output)
            state_ta = nest.map_structure(lambda ta, state: ta.write(time, state), state_ta, next_state)

            elements_finished = math_ops.logical_or(elements_finished, next_finished)

            return (next_time, elements_finished, next_input, state_ta,
                    emit_ta, next_state, loop_state)

        returned = control_flow_ops.while_loop(
            condition, body, loop_vars=[
                time, elements_finished, next_input, state_ta,
                emit_ta, state, loop_state],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory
        )

        (state_ta, emit_ta, final_state, final_loop_state) = returned[-4:]

        flat_states = nest.flatten(state_ta)
        flat_states = [array_ops.transpose(ta.stack(), (1, 0, 2)) for ta in flat_states]
        states = nest.pack_sequence_as(structure=state_ta, flat_sequence=flat_states)

        flat_outputs = nest.flatten(emit_ta)
        flat_outputs = [array_ops.transpose(ta.stack(), (1, 0, 2)) for ta in flat_outputs]
        outputs = nest.pack_sequence_as(structure=emit_ta, flat_sequence=flat_outputs)

        return (states, outputs, final_state)


def rnn_teacher_force(inputs, cell, sequence_length, initial_state, scope='dynamic-rnn-teacher-force'):
    """
    Implementation of an rnn with teacher forcing inputs provided.
    Used in the same way as tf.dynamic_rnn.
    """
    inputs = array_ops.transpose(inputs, (1, 0, 2))
    inputs_ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
    inputs_ta = inputs_ta.unstack(inputs)

    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        next_cell_state = initial_state if cell_output is None else cell_state

        elements_finished = time >= sequence_length
        finished = math_ops.reduce_all(elements_finished)

        next_input = control_flow_ops.cond(
            finished,
            lambda: array_ops.zeros([array_ops.shape(inputs)[1], inputs.shape.as_list()[2]], dtype=dtypes.float32),
            lambda: inputs_ta.read(time)
        )

        next_loop_state = None
        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    states, outputs, final_state = raw_rnn(cell, loop_fn, scope=scope)
    return states, outputs, final_state


def rnn_free_run(cell, initial_state, sequence_length, initial_input=None, scope='dynamic-rnn-free-run'):
    """
    Implementation of an rnn which feeds its feeds its predictions back to itself at the next timestep.

    cell must implement two methods:

        cell.output_function(state) which takes in the state at timestep t and returns
        the cell input at timestep t+1.

        cell.termination_condition(state) which returns a boolean tensor of shape
        [batch_size] denoting which sequences no longer need to be sampled.
    """
    with vs.variable_scope(scope, reuse=True):
        if initial_input is None:
            initial_input = cell.output_function(initial_state)

    def loop_fn(time, cell_output, cell_state, loop_state):
        next_cell_state = initial_state if cell_output is None else cell_state

        elements_finished = math_ops.logical_or(
            time >= sequence_length,
            cell.termination_condition(next_cell_state)
        )
        finished = math_ops.reduce_all(elements_finished)

        next_input = control_flow_ops.cond(
            finished,
            lambda: array_ops.zeros_like(initial_input),
            lambda: initial_input if cell_output is None else cell.output_function(next_cell_state)
        )
        emit_output = next_input[0] if cell_output is None else next_input

        next_loop_state = None
        return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)

    states, outputs, final_state = raw_rnn(cell, loop_fn, scope=scope)
    return states, outputs, final_state
