import tensorflow as tf


class GAN(object):

    @classmethod
    def make_model(cls, output_dim, init='ones', activation='linear', hidden_layers=[32], dropout=0., output_act='tanh', name="generator"):
        layers = cls.make_layers(init, hidden_layers, activation, dropout)
        layers.append(tf.keras.layers.Dense(output_dim, activation=output_act, kernel_initializer=init))
        return tf.keras.Sequential(layers, name=name)

    @classmethod
    def make_layers(cls, init='ones', hidden_layers=[32], activation='linear', dropout=0.):
        layers = []
        for n in hidden_layers:
            layers.append(tf.keras.layers.Dense(n, activation=activation, kernel_initializer=init))
            if dropout > 0:
                layers.append(tf.keras.layers.Dropout(dropout))
        return layers


class FairGAN(tf.keras.Model):

    def __init__(self, metrics, **params):
        super(FairGAN, self).__init__()

        self.ranker_dis_step = params['ranker_dis_step']
        self.ranker_gen_step = params['ranker_gen_step']
        self.controller_dis_step = params['controller_dis_step']
        self.controller_gen_step = params['controller_gen_step']
        self.controlling_fairness_step = params['controlling_fairness_step']

        self.ranker_initailizer = params['ranker_initializer']
        self.controller_initailizer = params['controller_initializer']
        self.alpha = params['alpha']
        self._lambda = params['lambda']
        self.n_items = params['n_items']

        self.ranker_gen_reg = params['ranker_gen_reg']
        self.ranker_dis_reg = params['ranker_dis_reg']
        self.controller_gen_reg = params['controller_gen_reg']
        self.controller_dis_reg = params['controller_dis_reg']
        self.controlling_fairness_reg = params['controlling_fairness_reg']

        self.create_models(params)
        
        # compile
        self.compile(metrics=metrics, params=params)


    def create_models(self, params):
        # Generator Models
        self.ranker_gen = GAN.make_model(self.n_items, 
                                    hidden_layers=params['ranker_gen_layers'], 
                                    activation=params['ranker_gen_activation'], 
                                    dropout=params['ranker_gen_dropout'], init=self.ranker_initailizer, output_act='tanh', name="ranker_gen")

        self.controller_gen = GAN.make_model(self.n_items, 
                                    hidden_layers=params['controller_gen_layers'], 
                                    activation=params['controller_gen_activation'], 
                                    dropout=params['controller_gen_dropout'], init=self.controller_initailizer, output_act='softmax', name="controller_gen")


        # Discriminator Models
        self.ranker_dis = GAN.make_model(1, hidden_layers=params['ranker_dis_layers'], 
                                        activation=params['ranker_dis_activation'], 
                                        dropout=params['ranker_dis_dropout'], init=self.ranker_initailizer, output_act='sigmoid', name="ranker_dis")
        
        self.controller_dis = GAN.make_model(1, hidden_layers=params['controller_dis_layers'], 
                                            activation=params['controller_dis_activation'], 
                                            dropout=params['controller_dis_dropout'], init=self.controller_initailizer, output_act='sigmoid', name="controller_dis")

        
        self.controller_dis.build(input_shape=(None, self.n_items))
        self.controller_gen.build(input_shape=(None, self.n_items))
        self.controller_dis.save_weights("controller_dis.h5")
        self.controller_gen.save_weights("controller_gen.h5")

    def compile(self, metrics=None, params=None):
        super(FairGAN, self).compile(weighted_metrics=metrics, run_eagerly=params['debug'])
        self.metrics = metrics
        
        # optimizers

        self.ranker_gen_optimizer = tf.keras.optimizers.Adam(learning_rate=params['ranker_gen_lr'], 
                                                              beta_1=params['ranker_gen_beta1'])    

        self.ranker_dis_optimizer = tf.keras.optimizers.Adam(learning_rate=params['ranker_dis_lr'], 
                                                            beta_1=params['ranker_dis_beta1'])                                               

        self.controller_gen_optimizer = tf.keras.optimizers.Adam(learning_rate=params['controller_gen_lr'], 
                                                              beta_1=params['controller_gen_beta1'])

        self.controller_dis_optimizer = tf.keras.optimizers.Adam(learning_rate=params['controller_dis_lr'], 
                                                       beta_1=params['controller_dis_beta1'])
        
        self.controlling_fairness_optimizer = tf.keras.optimizers.Adam(learning_rate=params['controlling_fairness_lr'], 
                                                              beta_1=params['controlling_fairness_beta1'])


    def gradient_penalty(self, discriminator, ground_truth, gen_fake_output, component):
        epsilon = tf.random.uniform([], 0.0, 1.0)
        x_hat = epsilon * ground_truth + (1 - epsilon) * gen_fake_output
        with tf.GradientTape() as t:
            t.watch(x_hat)
            if component == "ranker":
                # ground_truth is same as conditions
                d_hat = discriminator(tf.concat([x_hat, ground_truth], axis=1))
            else:
                d_hat = discriminator(x_hat)

        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=0) + 1e-12)
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    def discriminator_loss(self, discriminator, ground_truth, gen_fake_output, dis_real_output, dis_fake_output, component="ranker"):
        d_regularizer = self.gradient_penalty(discriminator, ground_truth, gen_fake_output, component)
        return -tf.reduce_mean(dis_real_output) + tf.reduce_mean(dis_fake_output) + d_regularizer * self._lambda

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def test_step(self, val_ds):
        conditions, labels = val_ds
        labels = tf.cast(labels, tf.float32)
        conditions = tf.cast(conditions, tf.float32)
        ranker_predictions = self.ranker_gen(conditions, training=False)
        ranker_generated_scores = tf.cast(ranker_predictions * (1 - conditions), tf.float32)

        rets = {}
        # Update the metrics.
        for m in self.metrics:
            rets[m.name] = m(labels, ranker_generated_scores)

        # Return a dict mapping metric names to current value.
        return rets

    def train_step(self, train_ds):
        conditions, labels = train_ds
        labels = tf.cast(labels, tf.float32)
        conditions = tf.cast(conditions, tf.float32)

        # Train Ranker Discriminator
        for _ in range(self.ranker_dis_step):
            with tf.GradientTape() as tape:

                # Decode input to fake item scores
                ranker_gen_pred = self.ranker_gen(conditions, training=False)
                fake_combined_inputs = tf.concat([ranker_gen_pred * conditions, conditions], axis=1)

                # Combine real item scores with user condition
                real_combined_inputs = tf.concat([labels, conditions], axis=1)

                ranker_dis_real_output = self.ranker_dis(real_combined_inputs, training=True)
                ranker_dis_fake_output = self.ranker_dis(fake_combined_inputs, training=True)

                loss = self.discriminator_loss(self.ranker_dis, labels, ranker_gen_pred * conditions, ranker_dis_real_output, ranker_dis_fake_output, component="ranker")
                loss += tf.add_n([ tf.nn.l2_loss(v) for v in self.ranker_dis.trainable_weights]) * self.ranker_dis_reg
                
            grads = tape.gradient(loss, self.ranker_dis.trainable_weights)
            self.ranker_dis_optimizer.apply_gradients(zip(grads, self.ranker_dis.trainable_weights))
            
        # Train Ranker Generator
        for _ in range(self.ranker_gen_step):
            with tf.GradientTape() as tape:

                # Decode input to fake item scores
                ranker_gen_pred = self.ranker_gen(conditions, training=True)

                # Distributor
                ranker_dis_fake_output = self.ranker_dis(tf.concat([ranker_gen_pred * conditions, conditions], axis=1), training=False)
                loss = self.generator_loss(ranker_dis_fake_output) 
                loss += tf.add_n([ tf.nn.l2_loss(v) for v in self.ranker_gen.trainable_weights]) * self.ranker_gen_reg

            grads = tape.gradient(loss, self.ranker_gen.trainable_weights)
            self.ranker_gen_optimizer.apply_gradients(zip(grads, self.ranker_gen.trainable_weights))

        # reset weights of controller each iteration
        self.controller_dis.load_weights("controller_dis.h5")
        self.controller_gen.load_weights("controller_gen.h5")

        # Train Controller Discriminator
        for _ in range(self.controller_dis_step):
            with tf.GradientTape() as tape:

                # Decode input to fake item scores
                ranker_gen_pred = self.ranker_gen(conditions, training=False)
                controller_gen_pred = self.controller_gen(ranker_gen_pred, training=False)

                controller_exposure_target = tf.cast(tf.argsort(tf.argsort(ranker_gen_pred * (1 - labels) + labels, direction="DESCENDING")), tf.float32) + 1.
                controller_exposure_target = 1. / tf.math.log(1. + controller_exposure_target)
                controller_exposure_target = tf.nn.softmax(controller_exposure_target)

                controller_dis_real_output = self.controller_dis(controller_exposure_target, training=True)
                controller_dis_fake_output = self.controller_dis(controller_gen_pred, training=True)

                loss = self.discriminator_loss(self.controller_dis, labels, controller_gen_pred, controller_dis_real_output, controller_dis_fake_output, component="controller")
                loss += tf.add_n([ tf.nn.l2_loss(v) for v in self.controller_dis.trainable_weights]) * self.controller_dis_reg

            grads = tape.gradient(loss, self.controller_dis.trainable_weights)
            self.controller_dis_optimizer.apply_gradients(zip(grads, self.controller_dis.trainable_weights))
            
        # Train Controller Generator
        for _ in range(self.controller_gen_step):
            with tf.GradientTape() as tape:

                # Decode input to fake item scores
                ranker_gen_pred = self.ranker_gen(conditions, training=False)
                controller_gen_pred = self.controller_gen(ranker_gen_pred, training=True)

                # Distributor
                controller_dis_fake_output = self.controller_dis(controller_gen_pred, training=False)
                loss = self.generator_loss(controller_dis_fake_output) 
                loss += tf.add_n([ tf.nn.l2_loss(v) for v in self.controller_gen.trainable_weights]) * self.controller_gen_reg

            grads = tape.gradient(loss, self.controller_gen.trainable_weights)
            self.controller_gen_optimizer.apply_gradients(zip(grads, self.controller_gen.trainable_weights))

        # Controling Fairness by Minimizing IED of e^hat
        for _ in range(self.controlling_fairness_step):
            with tf.GradientTape() as tape:

                # Decode input to fake item scores
                ranker_gen_pred = self.ranker_gen(conditions, training=True)
                controller_gen_pred = self.controller_gen(ranker_gen_pred, training=False)

                approx_exposure = tf.reduce_mean(controller_gen_pred, axis=0)
                loss = self.alpha * tf.reduce_sum(tf.abs(approx_exposure[:, None] - approx_exposure[None, :])) / (2. * tf.cast(self.n_items, tf.float32) * tf.reduce_sum(approx_exposure))
                loss += tf.add_n([ tf.nn.l2_loss(v) for v in self.ranker_gen.trainable_weights]) * self.controlling_fairness_reg

            grads = tape.gradient(loss, self.ranker_gen.trainable_weights)
            self.controlling_fairness_optimizer.apply_gradients(zip(grads, self.ranker_gen.trainable_weights))
        return {}
