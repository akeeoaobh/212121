# -*- coding: utf-8 -*-
#
"""
Created on Sat Oct 14 17:23:31 2017

@author: Garma
"""

import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)#建立 tensor的constant <定值初始宣告了就不能改變>
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)#印出來 但沒有值 需經過計算(evaluate) 只有屬性
#Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
sess = tf.Session()#要創建一個session來讓他進行evaluate
print(sess.run([node1, node2]))
#[3.0, 4.0]
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))


a = tf.placeholder(tf.float32)#A placeholder is a promise to provide a value later
b = tf.placeholder(tf.float32)#placeholder=保留一個位置
print(a, b)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))


W = tf.Variable([.3], dtype=tf.float32)#variable 可以變更值
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b #這是一個線性模組
init = tf.global_variables_initializer()#無法在一開始定義代數的時候宣告 必須要呼叫這一行 並run她才算初始化完成
sess.run(init)

