# -*- coding: utf-8 -*-
#
"""
Created on Sat Oct 14 17:23:31 2017

@author: Garma
"""

import tensorflow as tf
import numpy as np
#建立 tensor的constant <定值初始宣告了就不能改變>
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)#印出來 但沒有值 需經過計算(evaluate) 只有屬性
#Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
sess = tf.Session()#要創建一個session來讓他進行evaluate
print(sess.run([node1, node2]))
#[3.0, 4.0]
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

#A placeholder is a promise to provide a value later
#placeholder=保留一個位置
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
print(a, b)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

##variable 可以變更值
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b #這是一個線性模型
init = tf.global_variables_initializer()#無法在一開始定義代數的時候宣告 必須要呼叫這一行 並run她才算初始化完成
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
#[ 0.          0.30000001  0.60000002  0.90000004]

#為了判斷我們模型的正確性 我們要寫一個loss function來計算這個模型結果和實際獲得結果的差異
#這邊用的是standard loss model for linear regression
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)#平方
loss = tf.reduce_sum(squared_deltas)#總和
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-1.])#Veriable可以用Assign變更
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


#tf.train API
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print(sess.run([W, b]))

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))