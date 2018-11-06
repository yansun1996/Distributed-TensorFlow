
# coding: utf-8 

# # Import
import tensorflow as tf

########################################################################################
# # TFRecord version graph

def tfrecord_model_graph(ids, values, labels, test_ids, test_values, test_labels, dim_feature, dim_label, auc_threshold, reload_model, test_option):
    '''
    Define model inference
    '''
    with tf.variable_scope("logistic_regression"):

        # reload_model does not require initializer
        if reload_model != "xxx":
            init_method = None
        else:
            init_method = tf.random_normal_initializer()
    
        with tf.name_scope("Weight"):
            W = tf.get_variable("weights", [dim_feature, dim_label], initializer=init_method)
        with tf.name_scope("Bias"):
            b = tf.get_variable("bias", [dim_label, dim_label], initializer=init_method)
        with tf.name_scope("Output"):
            output = tf.nn.embedding_lookup_sparse(W, ids, values, combiner="sum") + b
        with tf.name_scope("Loss"):
            labels = tf.to_int64(labels)
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=tf.cast(tf.reshape(labels, [-1,1]), tf.float32)))
        with tf.name_scope("AUC"):
            labels = tf.to_int64(labels)
            # Use auc_update_op as the parameter in sess.run() to calculate auc
            # Use auc as return value for getting auc value
            auc, auc_update_op = tf.metrics.auc(
                #labels=tf.cast(tf.reshape(labels,[-1,1]),tf.bool),
                #predictions=tf.sigmoid(output),
                labels=tf.one_hot(labels, depth=2),
                predictions=tf.concat([1-tf.sigmoid(output),tf.sigmoid(output)],axis=1),
                weights=None,
                num_thresholds=auc_threshold,
                metrics_collections=None,
                updates_collections=None,
                curve='ROC',
                name=None,
                #summation_method='trapezoidal'
            )

        if test_option:
            with tf.name_scope("Test_Output"):
                test_output = tf.nn.embedding_lookup_sparse(W, test_ids, test_values, combiner="sum") + b
            with tf.name_scope("Test_Loss"):
                test_labels = tf.to_int64(test_labels)
                test_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=test_output, labels=tf.cast(tf.reshape(test_labels, [-1,1]), tf.float32)))
            with tf.name_scope("Test_AUC"):
                test_auc, test_auc_update_op = tf.metrics.auc(
                    #labels=tf.cast(tf.reshape(test_labels,[-1,1]),tf.bool),
                    #predictions=tf.sigmoid(test_output),
                    labels=tf.one_hot(test_labels, depth=2),
                    predictions=tf.concat([1-tf.sigmoid(test_output),tf.sigmoid(test_output)],axis=1),
                    weights=None,
                    num_thresholds=auc_threshold,
                    metrics_collections=None,
                    updates_collections=None,
                    curve='ROC',
                    name=None, 
                    #summation_method='trapezoidal'
                    )

        tf.summary.scalar("Loss", cost)
        tf.summary.scalar("Auc", auc)

        if test_option:
            tf.summary.scalar("Test_Loss", test_cost)
            tf.summary.scalar("Test_Auc", test_auc)

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

    if test_option:
        return W, b, cost, auc_update_op, test_cost, test_auc, test_auc_update_op, summary_op, saver
    else:
        return W, b, cost, auc_update_op, summary_op, saver
########################################################################################
# # Calc Average Gradient for Multuple gpu

def average_gradients(tower_grads):
   average_grads = []
   for grad_and_vars in zip(*tower_grads):
       grads = []
       for g, var in grad_and_vars:
           if g is None:
               g = tf.zeros_like(var)
           expanded_g = tf.expand_dims(g, 0)
           grads.append(expanded_g)
       grad = tf.concat(axis=0, values=grads)
       grad = tf.reduce_mean(grad, 0)
       v = grad_and_vars[0][1]
       grad_and_var = (grad, v)
       average_grads.append(grad_and_var)
   return average_grads
