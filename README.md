# 2021-AIAC-QQ-Browser-Hyperparameter-Optimization-Rank6

2021 AIAC QQ浏览器AI算法大赛 赛道二 超参数优化 初赛Rank3 决赛Rank6



# 赛题官网：https://algo.browser.qq.com/

赛题内容：在信息流推荐业务场景中普遍存在模型或策略效果依赖于“超参数”的问题，而“超参数"的设定往往依赖人工经验调参，不仅效率低下维护成本高，而且难以实现更优效果。因此，本次赛题以超参数优化为主题，从真实业务场景问题出发，并基于脱敏后的数据集来评测各个参赛队伍的超参数优化算法。本赛题为超参数优化问题或黑盒优化问题：给定超参数的取值空间，每一轮可以获取一组超参数对应的Reward，要求超参数优化算法在限定的迭代轮次内找到Reward尽可能大的一组超参数，最终按照找到的最大Reward来计算排名。

算法baseline主要来自华为HEBO，针对比赛做了一些参数和代码的修改。另外官方提供的代码修改了一些结构方便线下debug。

运行环境: win10 ,Python3.6,Pycharm20200101,git bash用于运行打包脚本。

# 官方代码主要修改点：

1、thpo/run_search.py函数，增加修改如下代码：

```
#run_cmd = common.PYTHONX + " ./thpo/run_search_one_time.py " + common.args_to_str(cur_args)
args = common.parse_args(common.experiment_parser("description"))
searcher_root = args[common.CmdArgs.searcher_root]
searcher = get_implement_searcher(searcher_root)
eva_func_list = args[common.CmdArgs.data]
repeat_num = args[common.CmdArgs.repear_num]
err_code, err_msg = run_search_one_time(args, searcher, eva_func_list[0], repeat_num)
```

2、初赛阶段，修改n_iteration为10次，总共50组参数，因为hebo线下很容易就到0.99+，将迭代的次数减小，方便继续优化，线下线上能保证同时上分。

# hebo代码修改点：

1、修改代码结构，适配本次比赛，具体可以查看searcher.py.

2、searcher.py，name='gpy',MACE方法改为MOMeanSigmaLCB，EvolutionOpt修改iters参数为25.决赛优化check_unique的去重代码。在获得一批最优点后，增加通过距离选择其中一些点的方法，优于hebo原代码中的随机选择方式。具体在distance相关代码。

3、bo/models/gp/gpy_wgp.py,Matern32改为Matern52，去掉linear核，optimize_restarts修改为原来的三分之一，restarts改为一次，也就是优化一次。

# 总结
上面是本次比赛初赛和决赛的一些修改点，其它的漏掉的记起来了再补充。因为之前没做过超参数的优化，所以除了读大量论文和代码花了很多时间，调参也是花了很多时间。所以try.txt里面记录了大量调参的过程和结果，留作记录。另外初赛阶段把NeurIPS 2020开源的代码都试了下，特别是turbo这个试了很久，感觉应该有效果，但是实际使用效果不佳。初赛阶段之所以做上面这些修改，主要原因是一开始hebo代码调通以后，线下0.99线上0.001，后面发现是超时问题，所以相关的调参工作基本上是优化代码的运行时间，确保精度不下降的情况下提高速度，最终逐步从0.7+优化到0.95+，不过初赛最终切榜的时候显示超时，线上分数掉到0.899+，rank3.

复赛阶段基本上代码没做太大修改，因为试了很多策略效果都不怎么理想。最终还是没用early stop策略。线上0.712+

reference里面有使用的相关开源代码的链接，里面也能找到相应的论文，细节部分可以看下论文里面。

# reference：

1、https://github.com/huawei-noah/HEBO/tree/master/HEBO

2、https://bbochallenge.com/leaderboard/

3、https://github.com/uber-research/TuRBO



