# EM model代码及注释如下：
```pythonsript
#coding=utf-8
# 导入EM算法所需要的库
from numpy import *
from scipy import stats
import time # 该模块提供了各种时间相关的函数
# 返回系统运行时间的精确值（以秒为单位），包括睡眠时间的长度。从而帮我们找出哪些部分需要优化，以便提高程序性能。
start = time.perf_counter()

# 第1大部分：实现EM算法的单次迭代过程
def em_single(priors,observations): ## em_single(priors, observations)函数的功能是模拟抛掷硬币实验并估计在一次迭代中，硬币A与硬币B正面朝上的概率。
 """
 EM算法的单次迭代
 Arguments
 ------------
 priors:[theta_A,theta_B]
 observation:[m X n matrix]

 Returns
 ---------------
 new_priors:[new_theta_A,new_theta_B]

 :param priors: 硬币A与硬币B正面朝上的概率的初始值，类型为list，如[0.6, 0.5]代表硬币A正面朝上的概率为0.6，硬币B正面朝上的概率为0.5。
 :param observations: 抛掷硬币的实验结果记录，类型为list。list 的行数代表做了几轮实验，列数代表每轮实验用某个硬币抛掷了几次。 list 中的值代表正反面，T (或0) 代表反面朝上，H (或1) 代表正面朝上。如 [[1, 0, 1]， [0, 1, 1]] 表示进行了两轮实验，每轮实验用某硬币抛掷三次。第一轮的结果是正反正，第二轮的结果是反正正。
 :return: 将估计出来的硬币A和硬币B正面朝上的概率组成list返回。如[0.4, 0.6]表示你认为硬币A正面朝上的概率为0.4，硬币B正面朝上的概率为0.6。
 """
 counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
 # 把硬币1用A表示，硬币2用B表示。
 # 这里利用directory作为可变容器存储 A 硬币和 B 硬币出现正反面的次数。
 theta_A = priors[0] # [0]表示取priors这个list中的第一个元素。这里用theta去定义未知量，通过迭代寻找其最优解。
 theta_B = priors[1] # [1]表示取priors这个list中的第二个元素
 #E step
 for observation in observations: #这里的一个observation对应的是用某个硬币连续抛掷多次的一组实验，例如下面observations
  len_observation = len(observation) # 函数len()返回括号中对象(此处为observation)的项目数
  num_heads = observation.sum()             # 正面次数（由于 1 代表正面朝上，所以求和之后的数字对应的即是硬币正面向上的次数）
  num_tails = len_observation-num_heads     # 反面次数
  # 扔硬币服从二项分布，这里利用二项分布概率求解公式分析 A硬币 或 B硬币 出现对应结果的期望值。
  # scipy.stats包中的binom类对象是表示二项分布的。其中pmf函数的语法是pmf(k, n, p)，对应P(X=k) = ((n!)/(k!(n-k)!)) * p^k * (1-p)^(n-k). n: 试验次数; k: n次试验中出现对应现象(此处为硬币正面朝上)的次数；p: 每次试验中出现对应现象的期望。
  contribution_A = stats.binom.pmf(num_heads,len_observation,theta_A)
  contribution_B = stats.binom.pmf(num_heads,len_observation,theta_B)
  # 将两个概率正规化，得到数据来自硬币A，B的概率
  # 上述stats.binom.pmf求出的数字很小，并且数字位数较多，通过正规化可以让数据更简洁。
  # 正则化之后的结果中，weight_A + weight_B = 1. 所以这里 weight_A 和 weight_B 的含义可以理解为本轮实验（对应observations中的一行）由 A 硬币掷出的概率和由 B 硬币掷出的概率。
  weight_A = contribution_A / (contribution_A + contribution_B)
  weight_B = contribution_B / (contribution_A + contribution_B)
  #更新在当前参数下A，B硬币产生的正反面次数：由于 weight_A 和 weight_B 可以理解为本轮实验由A或B硬币掷出的概率，所以这里 "weight_A * num_heads" 可以理解为 A 硬币或 B 硬币对正面朝上次数的贡献。
  #在每个循环中，分析一轮抛掷硬币的实验（即对应于observations中的一行内容）。
  counts['A']['H'] += weight_A * num_heads #利用counts['A']从counts这个directory中取值，通过+=运算符对相应的key进行赋值："x += i 等同于 x = x + i"
  counts['A']['T'] += weight_A * num_tails
  counts['B']['H'] += weight_B * num_heads
  counts['B']['T'] += weight_B * num_tails
  # 之后循环上述命令，直到计算完observations中的所有行。

 # M step
 # 利用上一步求得的期望值重新计算 new_theta_A 和 new_theta_B
 new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
 new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
 return [new_theta_A,new_theta_B]

# 第2大部分：实现EM算法的主循环
def em(observations,prior,tol = 1e-6,iterations=10000): ## em(observations, thetas, tol=1e-4, iterations=100)函数需要完成的功能是模拟抛掷硬币实验并迭代估计硬币A与硬币B正面朝上的概率。
 """
 EM算法
 ：param observations :观测数据
 ：param prior：模型初值。代表硬币A与硬币B正面朝上的概率的初始值，类型为list ，如下文的[0.6, 0.5]代表硬币A正面朝上的概率为0.6 ，硬币B正面朝上的概率为 0.5
 ：param tol：迭代结束阈值。代表差异容忍度，即当EM算法估计出来的参数prior不怎么变化时，可以提前挑出循环。上述容忍度为 1e-6 ，则表示若这次迭代的估计结果与上一次迭代的估计结果之间的L1距离小于1e-6则跳出循环。
 ：param iterations：最大迭代次数。
 ：return：局部最优的模型参数
 """
 iteration = 0;
 while iteration < iterations:
  new_prior = em_single(prior,observations)  #利用EM算法的单次迭代更新硬币A与硬币B正面朝上的概率。
  delta_change = abs(prior[0]-new_prior[0])
# 当差异小于tol（即算法收敛到一定精度），跳出循环，结束算法，否则继续迭代
  if delta_change < tol:
   break
  else:
   prior = new_prior
   iteration +=1
 return [new_prior,iteration]
# 这里给定了循环的两个终止条件：① 模型参数变化小于阈值(tol)；② 循环达到最大次数。

#硬币投掷结果
#用两种不同的硬币采样做了5次试验，每次试验丢10次，正反面结果表示在observations中，把observations写成矩阵。
observations = array([[1,0,0,0,1,1,0,1,0,1],
      [1,1,1,1,0,1,1,1,1,1],
      [1,0,1,1,1,1,1,0,1,1],
      [1,0,1,0,0,0,1,1,0,0],
      [0,1,1,1,0,1,1,1,0,1]])
#调用EM算法
print (em(observations,[0.6,0.5]))
end = time.perf_counter()
print('Running time: %f seconds'%(end-start))
```
# 测试结果如下
```
"C:\python 3.6\python.exe" "C:\Users\杨、\PycharmProjects\pythonProject\EM model.py" 
[[0.7967887593831098, 0.5195839356752803], 14]
Running time: 0.048189 seconds

Process finished with exit code 0
```
