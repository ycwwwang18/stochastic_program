import numpy as np

from pha import *


class Rolling:
    def __init__(self, arrival_rate_1, arrival_rate_2, n_physician, arrival_horizon, planning_horizon, n_sample, epsilon, probi):
        self.lambda_1 = arrival_rate_1
        self.lambda_2 = arrival_rate_2
        self.n_physician = n_physician  # TODO:可以修改医生数量
        self.arrival_horizon = arrival_horizon
        self.planning_horizon = planning_horizon
        self.n_sample = n_sample  # the number of scenario samples
        self.epsilon = epsilon  # PHA terminal threshold
        self.p = probi  # the probability of patient health status
        self.total_capacity = 480  # the total daily capacity of each physician
        # TODO: 容量减少 capacity open要对应调整
        self.online_capacities = np.array([  # online capacities of four physicians in the planning horizon
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60],
            [60, 60, 60, 60]
        ])
        self.online_capacities = self.online_capacities.T
        self.offline_capacities = np.array([  # offline capacities of four physicians in the planning horizon
            [240, 240, 240, 240],
            [240, 240, 240, 240],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420],
            [420, 420, 420, 420]
        ])
        self.offline_capacities = self.offline_capacities.T
        self.offline_open = np.array([  # whether offline capacities are open for each physician
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        self.offline_open = self.offline_open.T
        self.online_open = np.array([  # whether online capacities are open for each physician
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        self.online_open = self.online_open.T
        self.service_duration_fv_with_rv = 60  # service duration for a FV, minutes
        self.service_duration_on_rv = 30
        self.service_duration_off_rv = 40  # service duration for offline revisit
        self.service_duration_fv_without_rv = 40

    def run(self):
        X_1 = np.zeros(shape=(self.n_physician, self.arrival_horizon+self.planning_horizon))
        X_2 = np.zeros(shape=(self.n_physician, self.arrival_horizon+self.planning_horizon))
        Y = np.zeros(shape=(self.n_physician, self.arrival_horizon+self.planning_horizon))
        Z = np.zeros(shape=(self.n_physician, self.arrival_horizon+self.planning_horizon))
        for k in range(self.arrival_horizon):
            I_1 = np.random.poisson(lam=self.lambda_1)  # generate the number of FV with RV patients
            I_2 = np.random.poisson(lam=self.lambda_2)  # generate the number of FV without RV patients
            scenario_samples = np.random.binomial(1, self.p, size=(self.n_sample, I_1))  # 对当天患者问诊结果进行采样
            pha = PHA(self.n_sample, k, self.epsilon, self.n_physician, self.arrival_horizon, self.planning_horizon, scenario_samples, self.offline_capacities, self.online_capacities)
            x_1, x_2 = pha.run(I_1, I_2)
            realize_scenario_ind = np.random.randint(0, self.n_sample)
            y, z, _ = pha.get_second_stage_solutions(realize_scenario_ind)
            self.update_capacity(x_1, x_2, y, z)
            X_1[:, k+1:k+self.planning_horizon+1] += x_1[:, k+1:k+self.planning_horizon+1]
            X_2[:, k+1:k+self.planning_horizon+1] += x_2[:, k+1:k+self.planning_horizon+1]
            Y[:, k+1:k+self.planning_horizon+1] += y[:, k+1:k+self.planning_horizon+1]
            Z[:, k+1:k+self.planning_horizon+1] += z[:, k+1:k+self.planning_horizon+1]

            self.plot_schedule(X_1, X_2, Y, Z)



    def update_capacity(self, x1, x2, y, z):
        self.offline_capacities = self.offline_capacities - x1*self.service_duration_fv_with_rv - x2*self.service_duration_fv_without_rv - y*self.service_duration_off_rv
        self.online_capacities = self.online_capacities - z*self.service_duration_on_rv


    def plot_schedule(self, X1, X2, Y, Z):
        label = ['FV with RV', 'FV without RV', 'Offline Revisit', 'Online Revisit']
        for i in range(self.n_physician):
            x1 = X1[i] * self.service_duration_fv_with_rv
            x2 = X2[i] * self.service_duration_fv_without_rv
            y = Y[i] * self.service_duration_off_rv
            z = Z[i] * self.service_duration_on_rv
            capacity_assign = np.array([x1, x2, y, z])
            capacity_assign = capacity_assign.T

            x = np.arange(capacity_assign.shape[0])  # 横坐标的索引
            bottom = np.zeros_like(capacity_assign[:, 0])  # 初始化底部高度为0

            for j in range(capacity_assign.shape[1]):
                plt.bar(x, capacity_assign[:, j], width=0.45, bottom=bottom, label=label[j])
                bottom += capacity_assign[:, j]  # 更新底部高度
            plt.legend()
            plt.xlabel('Planning horizon(day)')
            plt.ylabel('Capacity(min)')
            plt.title(f'Physician {i+1}')
            plt.show()


if __name__ == "__main__":
    lam_1 = 60
    lam_2 = 50
    n_phy = 4
    arrival_h = 5
    planning_h = 12
    n_sam = 10
    eps = 0.1
    p = 0.4
    roll = Rolling(lam_1, lam_2, n_phy, arrival_h, planning_h, n_sam, eps, p)
    roll.run()



