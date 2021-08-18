import tensorflow as tf

# Gate for GTrXL
def gate_func(output, inp, d_model, kernel_initializer, gate='plus'):

  if gate == 'plus':
    return output + inp

  elif gate == 'input':
    # \sig(W_g^l.*x)*x + y
    _inp = tf.layers.dense(inp, d_model, use_bias=False,
                           activation=tf.nn.sigmoid,
                           kernel_initializer=kernel_initializer,
                           name='gate')
    return tf.multiply(_inp, inp) + output

  elif gate == 'output':
    # x + \sig(W_g^l.*x -b_g^l)*y
    _inp = tf.layers.dense(inp, d_model,
                           activation=tf.nn.sigmoid,
                           kernel_initializer=kernel_initializer,
                           name='gate')
    return inp + tf.multiply(_inp, output)

  elif gate == 'highway':
    #\sig(W_g^l.*x+b_g^l)*x + (1-\sig(W_g^l.*x+b_g^l))*y
    _inp = tf.layers.dense(inp, d_model,
                           activation=tf.nn.sigmoid,
                           kernel_initializer=kernel_initializer,
                           name='gate')
    return tf.multiply(_inp, inp) + tf.multiply(1-_inp, output)

  elif gate == 'sigmoid-tanh':
    # x + \sig(W_g^l.*x-b_g^l)*\tanh(U_g^l.*y)
    _inp = tf.layers.dense(inp, d_model,
                           activation=tf.nn.sigmoid,
                           kernel_initializer=kernel_initializer,
                           name='gate_W')
    _output = tf.layers.dense(output, d_model, use_bias=False,
                              activation=tf.nn.tanh,
                              kernel_initializer=kernel_initializer,
                              name='gate_U')
    return inp + tf.multiply(_inp, _output)

  elif gate == 'gru':
    r = tf.layers.dense(tf.concat([output, inp], axis=-1), d_model,
                        use_bias=False,
                        activation=tf.nn.sigmoid,
                        kernel_initializer=kernel_initializer,
                        name='gate_r')
    z = tf.layers.dense(tf.concat([output, inp], axis=-1), d_model,
                        activation=tf.nn.sigmoid,
                        kernel_initializer=kernel_initializer,
                        name='gate_z')
    h = tf.layers.dense(tf.concat([output, tf.multiply(r, inp)], axis=-1),
                        d_model, use_bias=False,
                        activation=tf.nn.tanh,
                        kernel_initializer=kernel_initializer,
                        name='gate_h')
    return tf.multiply(1-z, inp) + tf.multiply(z, h)


def positional_embedding(pos_seq, inv_freq, bsz=None):
  sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
  pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
  if bsz is not None:
    return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
  else:
    return pos_emb[:, None, :]


def positionwise_FF(inp, d_model, d_inner, dropout, kernel_initializer,
                    scope='ff', is_training=True, pre_lnorm=False, gate='plus'):

  with tf.variable_scope(scope):

    if pre_lnorm:
      output = tf.contrib.layers.layer_norm(inp, begin_norm_axis=-1)
    else:
      output = inp

    output = tf.layers.dense(output, d_inner, activation=tf.nn.relu,
                             kernel_initializer=kernel_initializer,
                             name='layer_1')
    output = tf.layers.dropout(output, dropout, training=is_training,
                               name='drop_1')
    output = tf.layers.dense(output, d_model,
                             kernel_initializer=kernel_initializer,
                             name='layer_2')
    output = tf.layers.dropout(output, dropout, training=is_training,
                               name='drop_2')
    if not pre_lnorm:
      output = tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1)
    else:
      output = gate_func(tf.nn.relu(output), inp, d_model,
                         kernel_initializer=kernel_initializer, gate=gate)

  return output


def rel_shift(x):
  x_size = tf.shape(x)

  x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
  x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
  x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
  x = tf.reshape(x, x_size)

  return x


def rel_multihead_attn(inp, r, r_w_bias, r_r_bias, attn_mask, mems, d_model,
                       n_head, d_head, dropout, dropatt, is_training,
                       kernel_initializer, scope='rel_attn', pre_lnorm=False,
                       gate='plus'):
  scale = 1 / (d_head ** 0.5)
  with tf.variable_scope(scope):

    if pre_lnorm:
      w = tf.contrib.layers.layer_norm(inp, begin_norm_axis=-1)
    else:
      w = inp

    #qlen = tf.shape(w)[0]
    #rlen = tf.shape(r)[0]
    #bsz = tf.shape(w)[1]
    qlen = w.get_shape().as_list()[0]
    rlen = r.shape[0]
    bsz = w.get_shape().as_list()[1]

    cat = tf.concat([mems, w],
                    0) if mems is not None and mems.shape.ndims > 1 else w
    w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False,
                              kernel_initializer=kernel_initializer, name='qkv')
    r_head_k = tf.layers.dense(r, n_head * d_head, use_bias=False,
                               kernel_initializer=kernel_initializer, name='r')

    w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
    w_head_q = w_head_q[-qlen:]

    #klen = tf.shape(w_head_k)[0]
    klen = w_head_k.shape[0]

    #w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
    #w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
    #w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])
    w_head_q = tf.reshape(w_head_q, [qlen, -1, n_head, d_head])
    w_head_k = tf.reshape(w_head_k, [klen, -1, n_head, d_head])
    w_head_v = tf.reshape(w_head_v, [klen, -1, n_head, d_head])

    r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

    rw_head_q = w_head_q + r_w_bias
    rr_head_q = w_head_q + r_r_bias

    AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
    BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
    BD = rel_shift(BD)

    attn_score = (AC + BD) * scale
    attn_mask_t = attn_mask[:, :, None, None]
    attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
    #size_t = tf.shape(attn_vec)
    size_t = attn_vec.shape
    attn_vec = tf.reshape(attn_vec, [size_t[0], -1, n_head * d_head])

    attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False,
                               kernel_initializer=kernel_initializer, name='o')
    attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)

    attn_out = tf.contrib.layers.layer_norm(attn_out, begin_norm_axis=-1)

    if not pre_lnorm:
      output = tf.contrib.layers.layer_norm(attn_out + inp, begin_norm_axis=-1)
    else:
      output = gate_func(tf.nn.relu(attn_out), inp, d_model,
                         kernel_initializer=kernel_initializer, gate=gate)

  return output


def _create_mask(qlen, mlen, same_length=False):
  attn_mask = tf.ones([qlen, qlen])
  mask_u = tf.matrix_band_part(attn_mask, 0, -1)
  mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
  attn_mask_pad = tf.zeros([qlen, mlen])
  ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
  if same_length:
    mask_l = tf.matrix_band_part(attn_mask, -1, 0)
    ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
  return ret

def _cache_mem(curr_out, prev_mem, mem_len=None):
  if mem_len is None or prev_mem is None:
    new_mem = curr_out
  elif mem_len == 0:
    return prev_mem
  else:
    new_mem = tf.concat([prev_mem, curr_out], 0)[- mem_len:]

  return tf.stop_gradient(new_mem)


def trxl(dec_inp, mems,
         n_layer=2,
         d_model=200, n_head=10, d_head=20, d_inner=200, mem_len=8,
         pre_lnorm=False, gate='plus',
         dropout=0.0, dropatt=0.0, is_training=True,
         seed=1,
         init='normal', # {normal|uniform}
         init_range=0.1,
         init_std=0.02,
         same_length=False, clamp_len=-1,
         untie_r=False,
         scope='transformer'):

  if init == 'uniform':
    initializer = tf.initializers.random_uniform(
          minval=-init_range,
          maxval=init_range,
          seed=seed)
  elif init == "normal":
    initializer = tf.initializers.random_normal(
          stddev=init_std,
          seed=seed)

  new_mems = []
  with tf.variable_scope(scope):
    if untie_r:
      r_w_bias = tf.get_variable('r_w_bias', [n_layer, n_head, d_head],
                               initializer=initializer)
      r_r_bias = tf.get_variable('r_r_bias', [n_layer, n_head, d_head],
                                 initializer=initializer)
    else:
      r_w_bias = tf.get_variable('r_w_bias', [n_head, d_head],
                                 initializer=initializer)
      r_r_bias = tf.get_variable('r_r_bias', [n_head, d_head],
                                 initializer=initializer)

    #qlen = tf.shape(dec_inp)[0]
    #mlen = tf.shape(mems[0])[0] if mems is not None else 0
    qlen = dec_inp.shape[0]
    mlen = mems[0].shape[0] if mems is not None else 0
    klen = mlen + qlen

    attn_mask = _create_mask(qlen, mlen, same_length)

    pos_seq = tf.range(klen - 1, -1, -1.0)
    if clamp_len > 0:
      pos_seq = tf.minimum(pos_seq, clamp_len)
    inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
    pos_emb = positional_embedding(pos_seq, inv_freq)

    output = tf.layers.dropout(dec_inp, dropout, training=is_training)
    pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

    if mems is None:
      mems = [None] * n_layer

    for i in range(n_layer):
      # cache new mems
      new_mems.append(_cache_mem(output, mems[i], mem_len))

      with tf.variable_scope('layer_{}'.format(i)):
        output = rel_multihead_attn(
            inp=output,
            r=pos_emb,
            r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
            r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
            attn_mask=attn_mask,
            mems=mems[i],
            d_model=d_model,
            n_head=n_head,
            d_head=d_head,
            dropout=dropout,
            dropatt=dropatt,
            is_training=is_training,
            kernel_initializer=initializer,
            pre_lnorm=pre_lnorm,
            gate=gate)
        output = positionwise_FF(
            inp=output,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            kernel_initializer=initializer,
            is_training=is_training,
            pre_lnorm=pre_lnorm,
            gate=gate)

    output = tf.layers.dropout(output, dropout, training=is_training)
    return tf.reshape(output, [-1, output.shape[-1]]), tf.stack(new_mems, axis=0)
