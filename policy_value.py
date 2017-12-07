import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def policy_value():
    with tf.variable_scope("policy_value"):
        state = tf.placeholder("float",[None,4])

        #newvals is future reward
        newvals = tf.placeholder("float",[None,1])
        
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])

        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[10,2])
        b2 = tf.get_variable("b2",[2])

        w3 = tf.get_variable("w3",[10,1])
        b3 = tf.get_variable("b3",[1])

        #policy gradient
        calculated = tf.matmul(h1,w2) + b2
        probabilities = tf.nn.softmax(calculated)

        actions = tf.placeholder("float",[None,2])
        advantages = tf.placeholder("float",[None,1])

        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss1 = -tf.reduce_sum(eligibility)
        
        #value gradient
        calculated1 = tf.matmul(h1,w3) + b3
        diffs = calculated1 - newvals
        loss2 = tf.nn.l2_loss(diffs)

        #policy loss + value loss
        loss = loss1+loss2

        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)#AdamOptimizer
        
        return probabilities,calculated1, actions,state,advantages, newvals, optimizer, loss1,loss2

def run_episode(env, policy_value, sess,is_train = True):    
    p_probabilities,v_calculated,p_actions, pv_state, p_advantages, v_newvals, pv_optimizer,loss1,loss2 = policy_value

    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []


    for _ in range(200):
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        #calculate action according to current state
        probs = sess.run(p_probabilities,feed_dict={pv_state: obs_vector})
        
        action = 1 if probs[0][0]<probs[0][1] else 0
        #take a random action when training
        if is_train:
            action = 0 if random.uniform(0,1) < probs[0][0] else 1
        # record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    #return totalreward if it is testing
    if not is_train:
        return totalreward
    
    #training
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        #value function: calculate max reward under current state 
        currentval = sess.run(v_calculated,feed_dict={pv_state: obs_vector})[0][0]

        # advantage: how much better was this action than normal
        # 根据实际数据得到future_reward比值函数计算出来的reward要好多少
        # 训练到后来,这个currentval:即在当前reward会估计的比较准确,在当前state下能够获得的
        # 最大reward或者平均reward,而有了这个估计,用实际的reward减去这个reward,就可以判断这个
        # action的好坏,即这个currentval是训练时用来评估某个action的好坏
        # 用future_reward减去这个最大reward,就得到了这个action
        # 对应的label,如果比估计的值更大,那说明要根据该参数进行更新,如果比该值小,那说明
        # 达不到平均水平,那么将将该action对应的梯度进行反向更新(相减为负值),使得下次碰到这个
        # 类似的state的时候,不再采取这个action
        advantages.append(future_reward - currentval)

        #advantages.append(future_reward-2.0)

        update_vals.append(future_reward)

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)

    advantages_vector = np.expand_dims(advantages, axis=1)
    #train network
    _,print_loss1,print_loss2 = sess.run([pv_optimizer,loss1,loss2], feed_dict={pv_state: states,v_newvals: update_vals_vector, p_advantages: advantages_vector, p_actions: actions})
    
    print("policy loss ",print_loss1)
    print("value loss ",print_loss2)
    return totalreward


env = gym.make('CartPole-v0')

PolicyValue = policy_value()

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

for i in range(1500):
    reward = run_episode(env, PolicyValue, sess)

t = 0
for _ in range(1000):
    #env.render()
    reward = run_episode(env, PolicyValue, sess,False)
    t += reward
print(t / 1000)

