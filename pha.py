import numpy as np

from subprob_wang import *
from subprob_without_lag import *
from multi_process import *


class PHA:
    def __init__(self, n_scenario, k, epsilon):
        # parameters initializing
        self.n_scenario = n_scenario  # number of scenarios
        self.k = k  # request day during arrival horizon
        self.epsilon = epsilon  # 算法终止阈值
        self.n_physician = 4  # the number of physicians
        self.K = 5  # arriving horizon
        self.L = 12  # planning horizon
        self.T = self.K + self.L

    def run(self, i_1, i_2):
        sample_health_status = np.random.binomial(1, 0.4, size=(self.n_scenario, i_1))  # 对当天患者问诊结果进行采样
        # 算法参数
        lagrangian_multipliers_1 = np.zeros(shape=(self.n_scenario, i_1, 4, self.T))
        lagrangian_multipliers_2 = np.zeros(shape=(self.n_scenario, i_2, 4, self.T))
        rho = [0.1, 0.1]
        beta_1 = 1.1
        beta_2 = 1.2

        # 获取初始 consensus parameters
        initial_problem = SubProblemInit(i_1, i_2, self.k)
        args = [(status,) for status in sample_health_status]
        initial_cons_multiprocess = Multiprocess()
        initial_cons_multi = initial_cons_multiprocess.work(initial_problem.run_model, args, 0)
        initial_cons_1 = np.array(initial_cons_multi[0]).mean(axis=0)
        initial_cons_2 = np.array(initial_cons_multi[1]).mean(axis=0)

        n = 1
        while True:
            cons_1 = []
            cons_2 = []
            if n == 1:
                prob = SubProblem(i_1, i_2, self.k, rho, initial_cons_1, initial_cons_2)
            else:
                prob = SubProblem(i_1, i_2, self.k, rho, cons_1, cons_2)
            solve_multiprocess = Multiprocess()
            args = zip(sample_health_status, lagrangian_multipliers_1, lagrangian_multipliers_2)
            results_multi = solve_multiprocess.work(prob.run_model, args, n)
            assign_fv_with_rv = results_multi[0]
            assign_fv_without_rv = results_multi[1]
            assign_off = results_multi[2]
            assign_on = results_multi[3]
            off_or_on = results_multi[4]
            x_1 = np.array(assign_fv_with_rv)
            x_2 = np.array(assign_fv_without_rv)
            cons_1 = x_1.mean(axis=0)
            cons_2 = x_2.mean(axis=0)

            # # 对每个子问题进行求解
            # for n in range(self.n_scenario):
            #     sub_problem = SubProblem(sample_revisit_status[n], lagrangian_multipliers_1[n],
            #                              lagrangian_multipliers_2[n], rho, i_1, i_2, self.k)
            #     opt_x1, opt_x2, _, _, _ = sub_problem.run_model()
            #     x_1.append(opt_x1)
            #     x_2.append(opt_x2)
            #
            # # 计算 consensus variable
            # x_1 = np.stack(x_1, axis=0)
            # x_2 = np.stack(x_2, axis=0)
            # x_1_hat = x_1.mean(axis=0)
            # x_2_hat = x_2.mean(axis=0)

            # 根据 consensus variable 更新算法参数
            delta_x_1 = x_1 - np.repeat(cons_1[np.newaxis, :], repeats=self.n_scenario, axis=0)  # s, i, j, t
            delta_x_2 = x_2 - np.repeat(cons_2[np.newaxis, :], repeats=self.n_scenario, axis=0)
            lagrangian_multipliers_1 = lagrangian_multipliers_1 + rho[0] * delta_x_1
            lagrangian_multipliers_2 = lagrangian_multipliers_2 + rho[1] * delta_x_2
            rho[0] *= beta_1
            rho[1] *= beta_2

            # 判断是否达到终止条件
            r_1 = np.sum(delta_x_1 ** 2, axis=(1,2,3))
            r_1 = r_1.mean(axis=0)
            r_2 = np.sum(delta_x_2 ** 2, axis=(1, 2, 3))
            r_2 = r_2.mean(axis=0)
            if r_1 <= self.epsilon and r_2 <= self.epsilon:
                break

            n += 1

        x_1_hat = np.round(cons_1).astype(int)
        x_2_hat = np.round(cons_2).astype(int)

        return x_1_hat, x_2_hat


if __name__ == "__main__":
    N_scenario = 100  # number of scenarios
    K = 0
    I_1 = np.random.poisson(lam=28.5)  # 根据泊松分布对患者到达数量进行采样
    I_2 = np.random.poisson(lam=18.1)
    Epsilon = 0.01
    pha = PHA(N_scenario, K, Epsilon)
    opt_x_1, opt_x_2 = pha.run(I_1, I_2)  # 不同scenario对解的影响不大，所以高度一致
