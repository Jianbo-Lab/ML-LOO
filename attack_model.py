"""
This tutorial shows how to generate adversarial examples
using C&W attack in white-box setting.
The original paper can be found at:
https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from keras.layers import Input
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.train import train
from cleverhans.utils_keras import KerasModelWrapper
import pickle as pkl
import foolbox
from keras import backend as K  

from cleverhans.attacks import CarliniWagnerL2, FastGradientMethod, SaliencyMapMethod, DeepFool, BasicIterativeMethod
from foolbox.attacks import SaliencyMapAttack

class Attack(object):
	def __init__(self, model, *args):
		self.model = model

	def attack(self, x, *args):
		raise NotImplementedError

class CW(Attack):
	def __init__(self, model, source_samples = 2, binary_search_steps = 5, cw_learning_rate = 5e-3, confidence = 0, attack_iterations = 1000, attack_initial_const = 1e-2):
		super(Attack, self).__init__()
		
		model_wrap = KerasModelWrapper(model.model)
		self.model = model_wrap
		self.sess = model.sess

		self.x = model.input_ph
		self.y = Input(shape=(model.num_classes,), dtype = 'float32')

		abort_early = True
		self.cw = CarliniWagnerL2(self.model, sess=self.sess)
		self.cw_params = {
			'binary_search_steps': binary_search_steps,
			"y": None,
			'abort_early': True,
			'max_iterations': attack_iterations,
			'learning_rate': cw_learning_rate ,
			'batch_size': source_samples,
			'initial_const': attack_initial_const ,
			'confidence': confidence,
			'clip_min': 0.0,
		}

	def attack(self, x, y = None):
		print(self.cw_params)
		adv = self.cw.generate_np(x, **self.cw_params)

		if y:
			eval_params = {'batch_size': 100}
			preds = self.model.get_logits(self.x)
			acc = model_eval(self.sess, self.x, self.y, preds, adv, y, args=eval_params)
			adv_success = 1 - acc
			print('The adversarial success rate is {}.'.format(adv_success))

		return adv





